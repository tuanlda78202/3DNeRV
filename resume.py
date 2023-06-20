from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from src.model.baseline import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import wandb
from torchsummary import summary
import os
from pytorch_msssim import ms_ssim, ssim
from src.evaluation.evaluation import save_checkpoint, resume_checkpoint
import random

os.environ["WANDB_SILENT"] = "true"

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 5
FRAME_INTERVAL = 6
CROP_SIZE = 640

dataset, dataloader = build_dataloader(
    name="uvghd30",
    data_path="/home/tuanlda78202/3ai24/data/ready.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae(bs=BATCH_SIZE, fi=FRAME_INTERVAL, c3d=True).cuda()
# print(summary(model, (3, FRAME_INTERVAL, CROP_SIZE, CROP_SIZE), batch_size=1))

optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    # db = 20 * torch.log10(255.0 / torch.sqrt(mse))
    db = -10 * torch.log10(mse)  # bcs normalized
    return round(db.item(), 2)


def psnr_batch(batch_pred, batch_gt, bs, fi):
    psnr_list = []

    for batch_idx in range(bs):
        for fi_idx in range(fi):
            psnr_list.append(
                psnr(batch_pred[batch_idx][fi_idx], batch_gt[batch_idx][fi_idx])
            )

    return sum(psnr_list) / len(psnr_list)


start_epoch, model, optimizer = resume_checkpoint(
    model, optimizer, "/home/tuanlda78202/ckpt/ready-epoch399.pth"
)


wandb.init(project="vmae-nerv3d-1ke")

model.eval()

for batch_idx, data in enumerate(dataloader):
    data = data.permute(0, 4, 1, 2, 3).cuda()
    output = model(data)

    # PSNR
    pred = output
    gt = data.permute(0, 2, 1, 3, 4)

    loss = F.mse_loss(pred, gt)
    psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)

    data = torch.mul(data, 255)
    output = torch.mul(output, 255)

    pred = output.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)
    gt = data.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)

    pred = pred.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()

    wandb.log(
        {
            "loss": loss.item(),
            "psnr": psnr_db,
            "pred": wandb.Video(pred, fps=6, format="mp4"),
            "gt": wandb.Video(gt, fps=6, format="mp4"),
        },
    )

    del pred, gt, output, data

wandb.finish()
