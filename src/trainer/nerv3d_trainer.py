import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)

import time
import random
import torch
from .base_trainer import BaseTrainer
from utils import inf_loop
from tqdm import tqdm
import wandb
import gc


class NeRV3DTrainer(BaseTrainer):
    """
    Trainer class
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
        super().__init__(
            model, criterion, metric_ftns, optimizer, config, dataset, data_loader
        )
        self.config = config
        self.dataset = dataset
        self.data_loader = data_loader
        self.metric_ftns = metric_ftns

        self.batch_size = self.config["dataloader"]["args"]["batch_size"]
        self.frame_interval = self.config["dataloader"]["args"]["frame_interval"]
        self.valid_period = self.config["trainer"]["valid_period"]

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch

        self.lr_scheduler = lr_scheduler

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        tqdm_batch = tqdm(
            iterable=self.data_loader,
            desc="Epoch {}".format(epoch),
            total=len(self.data_loader),
            unit="it",
        )
        self.model.train()

        for batch_idx, data in enumerate(tqdm_batch):
            # BTHWC to BCTHW
            data = data.permute(0, 4, 1, 2, 3).cuda()
            pred = self.model(data)

            # BCTHW to BTCHW
            data = data.permute(0, 2, 1, 3, 4)
            loss = self.criterion(pred, data)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            self.lr_scheduler.step()

            # Metrics
            psnr = self.metric_ftns(
                pred,
                data,
                batch_size=self.batch_size,
                frame_interval=self.frame_interval,
            )

            tqdm_batch.set_postfix(loss=loss.item(), psnr=psnr)
            wandb.log(
                {
                    "loss": loss.item(),
                    "psnr": psnr,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
            )

            del data, pred, loss, psnr
            gc.collect()
            torch.cuda.empty_cache()

            if batch_idx == self.len_epoch:
                break

        tqdm_batch.close()

        if (epoch + 1) % self.valid_period == 0:
            self._valid_epoch(self)

    @staticmethod
    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: WandB log video that contains information about validation
        """
        self.model.eval()

        with torch.no_grad():
            random_num = random.randint(1, len(self.dataset))

            valid_data = (
                self.dataset[random_num].unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
            ).cuda()
            valid_pred = self.model(valid_data)

            valid_data = torch.mul(valid_data, 255).type(torch.uint8)
            valid_pred = torch.mul(valid_pred, 255).type(torch.uint8)

            # TCHW
            valid_data = (
                valid_data.permute(0, 2, 1, 3, 4).squeeze(0).cpu().detach().numpy()
            )
            valid_pred = valid_pred.squeeze(0).cpu().detach().numpy()

            wandb.log(
                {
                    "pred": wandb.Video(valid_pred, fps=4, format="mp4"),
                    "data": wandb.Video(valid_data, fps=4, format="mp4"),
                }
            )

            del valid_data, valid_pred
            gc.collect()
            torch.cuda.empty_cache()
