from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from src.model.hnerv3d import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import wandb
from torchsummary import summary
import os
from pytorch_msssim import ms_ssim, ssim
from src.evaluation.evaluation import save_checkpoint, resume_checkpoint
from src.evaluation.metric import *

import random


# os.environ["WANDB_SILENT"] = "true"

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 3
FRAME_INTERVAL = 4

dataset, dataloader = build_dataloader(
    name="uvghd30",
    data_path="data/beauty.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=(720, 1280),
)

# Model
model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=4,
    embed_dim=8,
    decode_dim=314,
    embed_size=(9, 16),
    scales=[5, 4, 2, 2],
    lower_width=6,
    reduce=3,
).cuda()

optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))

start_epoch, model, optimizer = resume_checkpoint(
    model, optimizer, "ckpt/checkpoint-epoch199.pth"
)


wandb.init(project="vmae-nerv3d-1ke", name="infer-flex3d-3M-beauty-hd-300e")

model.eval()

for batch_idx, data in enumerate(dataloader):
    data = data.permute(0, 4, 1, 2, 3).cuda()
    output = model(data)

    # PSNR
    pred = output
    gt = data.permute(0, 2, 1, 3, 4)

    loss = F.mse_loss(pred, gt)
    psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)

    data = torch.mul(data, 255).type(torch.uint8)
    output = torch.mul(output, 255).type(torch.uint8)

    pred = output.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1280)
    gt = data.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1280)

    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    wandb.log(
        {
            "loss": loss.item(),
            "psnr": psnr_db,
            "pred": wandb.Video(pred, fps=FRAME_INTERVAL, format="mp4"),
            "gt": wandb.Video(gt, fps=FRAME_INTERVAL, format="mp4"),
        },
    )

    del pred, gt, output, data

wandb.finish()
