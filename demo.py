from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from src.model.baseline import HNeRVMae
import torch
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)

# DataLoader
BATCH_SIZE = 1
FRAME_INTERVAL = 12
CROP_SIZE = 224
dataloader = build_dataloader(
    name="uvghd30",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae().cuda()

optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

# Training
for ep in range(10):
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
        pred = output.reshape(BATCH_SIZE, 3, FRAME_INTERVAL, CROP_SIZE, CROP_SIZE)
        target = data
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad()

        # Optimizer
        loss.backward()

        optimizer.step()

        tqdm_batch.set_postfix(loss=loss.item())
