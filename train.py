from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from src.model.baseline import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import wandb
from torchsummary import summary
import os
from src.evaluation.evaluation import save_checkpoint
from src.evaluation.metric import *

# os.environ["WANDB_SILENT"] = "true"

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 5
FRAME_INTERVAL = 4
CROP_SIZE = 960

dataset, dataloader = build_dataloader(
    name="uvghd30",
    data_path="data/beauty.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae(bs=BATCH_SIZE, fi=FRAME_INTERVAL, c3d=True).cuda()
# print(summary(model, (3, FRAME_INTERVAL, 720, 1080), batch_size=1))

start_epoch = 0
num_epoch = 400
learning_rate = 1e-3

optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epoch * len(dataset) / BATCH_SIZE, eta_min=1e-6
)

wandb.init(
    project="vmae-nerv3d-1ke",
    name="beauty720p-400e",
    config={
        "learning_rate": learning_rate,
        "epochs": num_epoch,
    },
)

# Training
for ep in range(start_epoch, num_epoch + 1):
    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Epoch {}".format(ep),
        total=len(dataloader),
        unit="it",
    )

    model.train()

    for batch_idx, data in enumerate(tqdm_batch):
        # BTHWC to BCTHW
        data = data.permute(0, 4, 1, 2, 3).cuda()

        # HNeRV MAE
        output = model(data)

        # Loss
        pred = output
        gt = data.permute(0, 2, 1, 3, 4)

        loss = F.mse_loss(pred, gt)
        psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)

        optimizer.zero_grad()

        # Optimizer
        loss.backward()

        optimizer.step()

        lrt = scheduler.get_last_lr()[0]
        tqdm_batch.set_postfix(loss=loss.item(), psnr=psnr_db, lr_scheduler=lrt)

        wandb.log({"loss": loss.item(), "psnr": psnr_db, "lr_scheduler": lrt})

        scheduler.step()

    if ep != 0 and (ep + 1) % 10 == 0:
        model.eval()

        data = next(iter(dataloader)).permute(0, 4, 1, 2, 3).cuda()
        output = model(data)

        data = torch.mul(data, 255)
        output = torch.mul(output, 255)

        pred = output.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1080)
        gt = data.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1080)

        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()

        wandb.log(
            {
                "pred": wandb.Video(pred, fps=6, format="mp4"),
                "gt": wandb.Video(gt, fps=6, format="mp4"),
            },
        )

        del pred, gt, output, data

    if ep != 0 and (ep + 1) % 100 == 0:
        save_checkpoint(ep, model, optimizer, loss)
        # resume_checkpoint(model, optimizer, resume_path)

wandb.finish()
