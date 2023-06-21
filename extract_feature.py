from src.dataset.build import build_dataloader
from tqdm import tqdm
from torch.optim import Adam
from src.model.baseline import HNeRVMae
import torch
import numpy as np
from src.evaluation.metric import *
import os
from src.evaluation.evaluation import resume_checkpoint
from tqdm import tqdm
import sys

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
    data_path="/home/tuanlda78202/3ai24/data/beauty.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# Model
model = HNeRVMae(bs=BATCH_SIZE, fi=FRAME_INTERVAL, c3d=True).cuda()
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))

start_epoch, model, optimizer = resume_checkpoint(
    model, optimizer, "/home/tuanlda78202/ckpt/beauty-epoch399.pth"
)

feature_list = []
save_path = "data/features"
url = os.path.join(save_path, "beauty.npy")

tqdm_batch = tqdm(
    iterable=dataloader,
    desc="Extract feature",
    total=len(dataloader),
    unit="it",
)
model.eval()

for batch_idx, data in enumerate(tqdm_batch):
    data = data.permute(0, 4, 1, 2, 3).cuda()
    model_feature = model.vmae_pretrained()
    feature = model_feature.forward_features(data)

    memory_feature = feature.element_size() * feature.nelement()
    memory_data = data.element_size() * data.nelement()

    tqdm_batch.set_postfix(
        memory_feat=memory_feature,
        memory_data=memory_data,
        feature_shape=feature.shape,
    )
    feature_list.append(feature.cpu().numpy())

# np.save(url, np.vstack(feature_list))
