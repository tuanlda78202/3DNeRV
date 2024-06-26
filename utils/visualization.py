import importlib
from datetime import datetime
from utils.util import init_wandb


class WandB:
    """
    Weights & Biases makes MLOps easy to track your experiments, manage & version your data,
    and collaborate with your team so you can focus on building the best models.
    """

    def __init__(self, name, cfg_trainer, logger, config, visual_tool="wandb"):
        self.writer = None
        self.selected_module = ""
        self.name = "wandb"

        if visual_tool != "None":
            # Retrieve visualization writer.
            succeeded = False

            # Import self.writer = wandb
            try:
                self.writer = importlib.import_module(visual_tool)
                succeeded = True

            except ImportError:
                succeeded = False

            self.selected_module = visual_tool

            # Install
            if not succeeded:
                message = (
                    "Warning: visualization (WandB) is configured to use, but currently not installed on this "
                    "machine. Please install WandB with 'pip install wandb', set the option in the 'config.yaml' file."
                )

                logger.warning(message)

        # Init writer based on WandB
        self.writer = init_wandb(
            self.writer,
            api_key_file=cfg_trainer["api_key_file"],
            mode=cfg_trainer["mode"],
            project=cfg_trainer["project"],
            entity=cfg_trainer["entity"],
            name=name,
            config=config,
        )

        self.step = 0
        self.mode = ""

        self.timer = datetime.now()
