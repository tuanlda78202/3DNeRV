import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from utils import load_yaml


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        Class to parse configuration yaml file. Handles hyper-parameters for training, initializations of modules, checkpoint saving
        and logging module.

        :param config: Dict containing configurations, hyper-parameters for training
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict key-chain:value, specifying position values to be replaced from config dict.
        """

        # Load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exp_name = self.config["trainer"]["name"]
        self._save_dir = save_dir / "models" / exp_name

        exist_ok = exp_name == ""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args, options=""):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.resume is not None:
            resume = Path(args.resume)
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yaml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None

        config = load_yaml(args.config)

        if args.config and resume:
            config.update(load_yaml(args.config))

        # parse custom cli options into dictionary
        modification = {
            opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options
        }

        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """

        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])

        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """

        module_name = self[name]["type"]

        if name in ("psnr", "msssim"):
            module_args = dict()
        else:
            module_args = dict(self[name]["args"])

        module_args.update(kwargs)

        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir


# Helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
