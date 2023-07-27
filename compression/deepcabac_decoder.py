from src.dataset.build import build_dataloader
from tqdm import tqdm
from src.backbone.videomaev2 import vit_small_patch16_224
import torch.nn.functional as F
from src.model.baseline import HNeRVMaeDecoder
import torch
import numpy as np
from src.evaluation.metric import *
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
    data_path="data/beauty.mp4",
    batch_size=BATCH_SIZE,
    frame_interval=FRAME_INTERVAL,
    crop_size=CROP_SIZE,
)

# VMAE
pretrained_mae = vit_small_patch16_224(all_frames=FRAME_INTERVAL)
checkpoint = torch.load("../vit_s_k710_dl_from_giant.pth", map_location="cpu")

for model_key in ["model", "module"]:
    if model_key in checkpoint:
        checkpoint = checkpoint[model_key]
        break

pretrained_mae.load_state_dict(checkpoint)
pretrained_mae.eval()
pretrained_mae.cuda()

for param in pretrained_mae.parameters():
    param.requires_grad = False

# Decoder Model
decoder_model = HNeRVMaeDecoder(fi=FRAME_INTERVAL).cuda()
state_dict = decoder_model.state_dict()

decoder = deepCABAC.Decoder()

with open("data/deepcabac/beauty.bin", "rb") as f:
    stream = f.read()

decoder.getStream(np.frombuffer(stream, dtype=np.uint8))

# param = decoder.decodeWeights()
# print(len(param[0][0]))

for name in tqdm(state_dict.keys()):
    param = decoder.decodeWeights()
    state_dict[name] = torch.tensor(param)

decoder.finish()

decoder_model.load_state_dict(state_dict)

# Inference
decoder_model.eval()

for batch_idx, data in enumerate(dataloader):
    data = data.permute(0, 4, 1, 2, 3).cuda()

    features = pretrained_mae.forward_features(data)
    output = decoder_model(features)

    # PSNR
    pred = output
    gt = data.permute(0, 2, 1, 3, 4)

    loss = F.mse_loss(pred, gt)
    psnr_db = psnr_batch(pred, gt, bs=BATCH_SIZE, fi=FRAME_INTERVAL)
    print(loss, psnr_db)

    del pred, gt, output, data
