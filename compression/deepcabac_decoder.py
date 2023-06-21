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
from pytorch_msssim import ms_ssim, ssim
from src.evaluation.evaluation import save_checkpoint, resume_checkpoint
import deepCABAC
import os

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
    data_path="/home/tuanlda78202/3ai24/data/bee.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae(bs=BATCH_SIZE, fi=FRAME_INTERVAL, c3d=True).cuda()
# print(summary(model, (3, FRAME_INTERVAL, 960, 960), batch_size=1))

optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))


decoder = deepCABAC.Decoder()

with open("data/beauty.bin", "rb") as f:
    stream = f.read()

decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
state_dict = model.state_dict()

for name in tqdm(state_dict.keys()):
    if ".num_batches_tracked" in name:
        continue
    param = decoder.decodeWeights()
    state_dict[name] = torch.tensor(param)
decoder.finish()

model.load_state_dict(state_dict)
