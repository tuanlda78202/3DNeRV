from src.dataset.datasets import VideoDataset
from src.backbone.videomaev2 import vit_small_patch16_224
from torch.utils.data import DataLoader
from timm.models import create_model
import torch
from tqdm import tqdm

# DataLoader
dataset = VideoDataset(
    anno_path="data/uvg_hd_30fps.csv", data_root="data", mode="validation"
)

beauty_data = dataset[0]
dataloader = DataLoader(beauty_data, batch_size=12)


# Extract feature

model = vit_small_patch16_224()
ckpt = torch.load("../vit_s_k710_dl_from_giant.pth", map_location="cpu")

for model_key in ["model", "module"]:
    if model_key in ckpt:
        ckpt = ckpt[model_key]
        break

model.load_state_dict(ckpt)
model.eval()
model.cuda()

tqdm_batch = tqdm(
    iterable=dataloader,
    desc="Epoch {}".format(0),
    total=len(dataloader),
    unit="it",
)

for batch_idx, data in enumerate(tqdm_batch):
    data = data.permute(3, 0, 1, 2).unsqueeze(0).cuda()

    # Forward feature ckpt
    with torch.no_grad():
        feature = model.forward_features(data)
