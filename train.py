from tqdm import tqdm
import numpy as np
import random
import wandb
import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from src.dataset.build import build_dataloader
from src.model.hnerv3d import HNeRVMae
from src.evaluation.metric import *
from src.evaluation.evaluation import save_checkpoint

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# CONFIG
BATCH_SIZE = 1
TRAIN_FRAME_INTERVAL = 4
DATA_PATH = "data/ready.mp4"
IMG_SIZE = (720, 1280)
START_EPOCH = 0
NUM_EPOCH = 300
LR = 1e-3
WB_NAME = "flex3d-3M-ready-hd-300e"

# Data
dataset, dataloader = build_dataloader(
    name="uvghd30",
    data_path=DATA_PATH,
    batch_size=BATCH_SIZE,
    frame_interval=TRAIN_FRAME_INTERVAL,
    crop_size=IMG_SIZE,
)

# Model
model = HNeRVMae(
    img_size=IMG_SIZE,
    frame_interval=TRAIN_FRAME_INTERVAL,
    embed_dim=8,
    decode_dim=314,
    embed_size=(9, 16),
    scales=[5, 4, 2, 2],
    lower_width=6,
    reduce=3,
).cuda()

compiled_model = torch.compile(model)

# Optimizer
optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCH * len(dataset) / BATCH_SIZE, eta_min=1e-6
)

wandb.init(project="vmae-nerv3d-1ke", name="WB_NAME")

# Training
for ep in range(START_EPOCH, NUM_EPOCH + 1):
    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Epoch {}".format(ep),
        total=len(dataloader),
        unit="it",
    )

    compiled_model.train()

    for batch_idx, data in enumerate(tqdm_batch):
        # BTHWC to BCTHW
        data = data.permute(0, 4, 1, 2, 3).cuda()
        output = compiled_model(data)

        pred = output
        gt = data.permute(0, 2, 1, 3, 4)  # BCTHW to BTCHW
        loss = F.mse_loss(pred, gt)

        psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=TRAIN_FRAME_INTERVAL)

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

        random_num = random.randint(0, len(dataset))

        data = dataset[random_num].unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
        pred = model(data)

        data = torch.mul(data, 255).type(torch.uint8)
        pred = torch.mul(pred, 255).type(torch.uint8)

        # TCHW
        data = data.permute(0, 2, 1, 3, 4).squeeze(0)
        pred = pred.squeeze(0)

        data = data.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        wandb.log(
            {
                "pred": wandb.Video(pred, fps=4, format="mp4"),
                "data": wandb.Video(data, fps=4, format="mp4"),
            },
        )
        del pred, gt, output, data, random_num

    if ep != 0 and (ep + 1) % 100 == 0:
        save_checkpoint(ep, model, optimizer, loss)
        # resume_checkpoint(model, optimizer, resume_path)

wandb.finish()
