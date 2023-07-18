from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from src.model.hnerv3d import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import wandb
from torchsummary import summary
import os
from src.evaluation.evaluation import save_checkpoint
from src.evaluation.metric import *
import time
import random
from src.dataset.yuv import YUVDataset

# os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 5
FRAME_INTERVAL = 4

dataset, dataloader = build_dataloader(
    name="uvg-raw",
    data_path="../uvg-raw/beauty.yuv",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=(720, 1280),
)

test_dataset = YUVDataset(
    data_path="../uvg-raw/beauty.yuv", frame_interval=10, crop_size=(720, 1280)
)

# Model
model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=FRAME_INTERVAL,
    encode_length=6300,
    embed_size=(36, 64),
    scales=[5, 2, 2],
).cuda()

# print(summary(model, (3, FRAME_INTERVAL, 720, 1080), batch_size=1))

test_model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=10,
    embed_size=(36, 32),
    scales=[5, 2, 2, 2],
).cuda()

start_epoch = 0
num_epoch = 400
learning_rate = 1e-3

optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epoch * len(dataset) / BATCH_SIZE, eta_min=1e-6
)

wandb.init(
    project="vmae-nerv3d-1ke",
    name="beauty-raw720p-400e",
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
        gt = data.permute(0, 2, 1, 3, 4)  # BCTHW to BTCHW
        loss = F.mse_loss(pred, gt)

        psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)

        optimizer.zero_grad()

        # Optimizer
        loss.backward()

        optimizer.step()

        lrt = scheduler.get_last_lr()[0]
        tqdm_batch.set_postfix(loss=loss.item(), psnr=psnr_db, lr_scheduler=lrt)

        # wandb.log({"loss": loss.item(), "psnr": psnr_db, "lr_scheduler": lrt})

        scheduler.step()

        del data, output, pred, gt, loss, psnr_db

    if ep != 0 and (ep + 1) % 10 == 0:
        model.eval()

        index_random = random.randint(0, len(dataset) - 1)
        test_data = test_dataset[index_random].permute(3, 0, 1, 2).unsqueeze(0).cuda()

        output = test_model(test_data)

        # BTCHW
        data = torch.mul(data, 255).permute(0, 2, 1, 3, 4).type(torch.uint8)
        output = torch.mul(output, 255).permute(0, 2, 1, 3, 4).type(torch.uint8)

        output = output.cpu().detach().numpy()
        data = data.cpu().detach().numpy()

        wandb.log(
            {
                "Prediction": wandb.Video(output, fps=4, format="mp4"),
                "GT": wandb.Video(data, fps=4, format="mp4"),
            },
        )

        del test_data, output, data

    if ep != 0 and (ep + 1) % 100 == 0:
        save_checkpoint(ep, model, optimizer, loss)
        # resume_checkpoint(model, optimizer, resume_path)

wandb.finish()
