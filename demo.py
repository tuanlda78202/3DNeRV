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

os.environ["WANDB_SILENT"] = "true"

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 1
FRAME_INTERVAL = 100
CROP_SIZE = 224
dataloader = build_dataloader(
    name="uvghd30",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae(bs=FRAME_INTERVAL).cuda()
# print(summary(model, (3, 12, 224, 224), batch_size=1))

optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))


def psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    # db = 20 * torch.log10(255.0 / torch.sqrt(mse))
    db = -10 * torch.log10(mse)  # bcs normalized
    return round(db.item(), 2)


def psnr_batch(batch_pred, batch_gt, bs):
    psnr_list = []

    for idx in range(bs):
        psnr_list.append(psnr(batch_pred[idx], batch_gt[idx]))

    return sum(psnr_list) / len(psnr_list)


wandb.init(project="baseline-hnerv-mae")

# Training
for ep in range(200):
    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Epoch {}".format(ep),
        total=len(dataloader),
        unit="it",
    )

    for batch_idx, data in enumerate(tqdm_batch):
        # BTHWC to BCTHW
        data = data.permute(0, 4, 1, 2, 3).cuda()

        # HNeRV MAE
        output = model(data)

        # Loss
        pred = output.reshape(FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)
        gt = data.reshape(FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)

        loss = F.mse_loss(pred, gt)
        psnr_db = psnr_batch(pred, gt, bs=FRAME_INTERVAL)

        optimizer.zero_grad()

        # Optimizer
        loss.backward()

        optimizer.step()

        tqdm_batch.set_postfix(loss=loss.item(), psnr=psnr_db)

        wandb.log({"loss": loss.item(), "psnr": psnr_db})

    if ep % 5 == 0:
        data = next(iter(dataloader)).permute(0, 4, 1, 2, 3).cuda()
        output = model(data)

        data = torch.mul(data, 255)
        output = torch.mul(output, 255)

        pred = output.reshape(FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)
        gt = data.reshape(FRAME_INTERVAL, 3, CROP_SIZE, CROP_SIZE)

        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()

        wandb.log(
            {
                "pred": wandb.Video(pred, fps=24, format="mp4"),
                "gt": wandb.Video(gt, fps=24, format="mp4"),
            },
        )

        wandb.log({"loss": loss.item(), "psnr": psnr_db})
        del pred, gt, output, data


wandb.finish()
