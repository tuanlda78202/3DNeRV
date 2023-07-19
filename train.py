from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
from src.model.hnerv3d import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import wandb
from src.evaluation.evaluation import save_checkpoint
from src.evaluation.metric import *

# os.environ["WANDB_MODE"] = "offline"
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

start_epoch = 0
num_epoch = 300
learning_rate = 1e-3

optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
scheduler = lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epoch * len(dataset) / BATCH_SIZE, eta_min=1e-6
)

wandb.init(project="vmae-nerv3d-1ke", name="flex3d-3M-beauty-hd-300e")

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

        wandb.log({"loss": loss.item(), "psnr": psnr_db, "lr_scheduler": lrt})

        scheduler.step()

    if ep != 0 and (ep + 1) % 10 == 0:
        model.eval()

        data = next(iter(dataloader)).permute(0, 4, 1, 2, 3).cuda()
        output = model(data)

        data = torch.mul(data, 255).type(torch.uint8)
        output = torch.mul(output, 255).type(torch.uint8)

        pred = output.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1280)
        gt = data.reshape(BATCH_SIZE, FRAME_INTERVAL, 3, 720, 1280)

        pred = pred.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()

        wandb.log(
            {
                "pred": wandb.Video(pred, fps=4, format="mp4"),
                "gt": wandb.Video(gt, fps=4, format="mp4"),
            },
        )

        del pred, gt, output, data

    if ep != 0 and (ep + 1) % 100 == 0:
        save_checkpoint(ep, model, optimizer, loss)
        # resume_checkpoint(model, optimizer, resume_path)

wandb.finish()
