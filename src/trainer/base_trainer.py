import torch
from abc import abstractmethod
from utils import WandB


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
        data_loader,
        lr_scheduler,
    ):
        self.config = config
        self.logger = config.get_logger("trainer", 1)

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        cfg_trainer = config["trainer"]
        self.name_exp = cfg_trainer["name"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.start_epoch = 0
        self.iters = 0

        self.checkpoint_dir = config.save_dir
        self.wandb = WandB(
            self.name_exp,
            cfg_trainer,
            self.logger,
            config=self.config,
        )

        if cfg_trainer["resume"]:
            self._resume_checkpoint(config.resume)

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

        self.wandb.writer.finish()

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
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

        filename = str(str(self.checkpoint_dir) + "/" + "{}e.pth".format(epoch + 1))

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
