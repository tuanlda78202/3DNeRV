import os
import sys
import wandb
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.getcwd())
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
from .base_trainer import BaseTrainer


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
        data_loader,
        lr_scheduler,
    ):
        super().__init__(
            model, criterion, metric_ftns, optimizer, config, data_loader, lr_scheduler
        )
        self.config = config
        self.data_loader = data_loader
        self.metric_ftns = metric_ftns
        self.lr_scheduler = lr_scheduler
        self.len_epoch = len(self.data_loader)

        self.frame_interval = config["dataloader"]["args"]["frame_interval"]
        self.valid_period = config["trainer"]["valid_period"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        tqdm_batch = tqdm(
            iterable=self.data_loader,
            desc="🚀 Epoch {}".format(epoch),
            total=self.len_epoch,
            unit="it",
        )

        self.model.train()
        train_loss_video, train_psnr_video = 0, 0

        for batch_idx, data in enumerate(tqdm_batch):
            # BTHWC > BCTHW
            data = data.permute(0, 4, 1, 2, 3)
            pred = self.model(data)

            # BCTHW > BTCHW
            data = data.permute(0, 2, 1, 3, 4)
            loss = self.criterion(pred, data)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # Metrics
            psnr = self.metric_ftns(pred, data)

            # TQDM & WandB
            tqdm_batch.set_postfix(
                lr=self.lr_scheduler.get_last_lr()[0], loss=loss.item(), psnr=psnr
            )

            wandb.log(
                {
                    "loss": loss.item(),
                    "psnr": psnr,
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
            )

            train_loss_video += loss.item()
            train_psnr_video += psnr

            if batch_idx == self.len_epoch:
                break

        tqdm_batch.close()

        wandb.log(
            {
                "Avg. Loss": train_loss_video / self.len_epoch,
                "Avg. PSNR": train_psnr_video / self.len_epoch,
            }
        )

        """
        print(
            "Train epoch {} | Avg. Loss: {:.4f} | Avg. PSNR: {:.4f}".format(
                epoch,
                train_loss_video / self.len_epoch,
                train_psnr_video / self.len_epoch,
            )
        )
        """

        if (epoch + 1) % self.valid_period == 0:
            self._valid_epoch(self, epoch)

    @staticmethod
    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: WandB log video that contains information about validation
        """

        valid_tqdm_batch = tqdm(
            iterable=self.data_loader,
            desc="🏆 Valid Epoch {}".format(epoch),
            total=self.len_epoch,
            unit="it",
        )

        self.model.eval()

        valid_loss_video, valid_psnr_video = 0, 0

        with torch.no_grad():
            for batch_idx, valid_data in enumerate(valid_tqdm_batch):
                # BTHWC > BCTHW
                valid_data = valid_data.permute(0, 4, 1, 2, 3)
                valid_pred = self.model(valid_data)

                # BCTHW > BTCHW
                valid_data = valid_data.permute(0, 2, 1, 3, 4)
                valid_loss = self.criterion(valid_pred, valid_data)

                # Metrics
                valid_psnr = self.metric_ftns(valid_pred, valid_data)

                # TQDM
                valid_tqdm_batch.set_postfix(
                    valid_loss=valid_loss.item(), valid_psnr=valid_psnr
                )

                # TCHW
                valid_data = torch.mul(valid_data, 255).type(torch.uint8)
                valid_pred = torch.mul(valid_pred, 255).type(torch.uint8)
                valid_data = valid_data.squeeze(0).cpu().detach().numpy()
                valid_pred = valid_pred.squeeze(0).cpu().detach().numpy()

                wandb.log(
                    {
                        "valid_loss": valid_loss.item(),
                        "valid_psnr": valid_psnr,
                        "valid_pred": wandb.Video(
                            valid_pred, fps=self.frame_interval, format="mp4"
                        ),
                        "valid_data": wandb.Video(
                            valid_data, fps=self.frame_interval, format="mp4"
                        ),
                    }
                )

                valid_loss_video += valid_loss.item()
                valid_psnr_video += valid_psnr

            valid_tqdm_batch.close()

            wandb.log(
                {
                    "avg_loss": valid_loss_video / self.len_epoch,
                    "avg_psnr": valid_psnr_video / self.len_epoch,
                }
            )

            print(
                "Valid epoch {} | Avg. Loss: {:.4f} | Avg. PSNR: {:.4f}".format(
                    epoch,
                    valid_loss_video / self.len_epoch,
                    valid_psnr_video / self.len_epoch,
                )
            )
