import torch
from abc import abstractmethod
from numpy import inf
from utils.log import WandB

"""
- Training process logging
- Checkpoint saving
- Checkpoint resuming
- Reconfigurable performance monitoring for saving current best model, and early stop training.
    - If config monitor is set to max val_accuracy, which means then the trainer will save a checkpoint model_best.pth 
      when validation accuracy of epoch replaces current maximum.
    - If config early_stop is set, training will be automatically terminated when model performance does not improve for given number of 
    epochs. This feature can be turned off by passing 0 to the early_stop option, or just deleting the line of config.
"""


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model,
        criterion,
        metric_ftns,
        optimizer,
        config,
        dataset,
        data_loader,
        lr_scheduler=None,
        len_epoch=None,
    ):
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.start_epoch = 0
        self.iters = 0
        self.checkpoint_dir = config.save_dir

        if cfg_trainer["resume"]:
            self._resume_checkpoint(config.resume)

        if cfg_trainer["visual_tool"] == "wandb":
            self.wandb = WandB(
                config["name"],
                cfg_trainer,
                self.logger,
                cfg_trainer["visual_tool"],
                config=self.config,
            )

        else:
            raise ImportError(
                "Visualization tool isn't exists, please refer to comment 1.* "
                "to choose appropriate module"
            )

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

            if epoch != 0 and (epoch + 1) % self.save_period == 0:
                self._save_checkpoint(epoch)

        self.wandb.finish()

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "iter": self.iters,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["arch"] != self.config["arch"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from that of "
                "checkpoint. This may yield an exception while state_dict is being loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["optimizer"]["type"]
            != self.config["optimizer"]["type"]
        ):
            self.logger.warning(
                "Warning: Optimizer type given in config file is different from that of checkpoint. "
                "Optimizer parameters not being resumed."
            )
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
