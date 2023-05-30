from src.dataset.datasets import VideoDataset
from src.backbone.videomaev2 import vit_small_patch16_224
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from math import ceil
import torch.nn.functional as F
from torch.optim import Adam

# DataLoader (just for one dataset)
dataset = VideoDataset(
    anno_path="data/uvghd_30fps.csv", data_root="data", mode="validation"
)
beauty_data = dataset[0]
beauty_dataloader = DataLoader(beauty_data, batch_size=12)

# Extract feature
cp_model = vit_small_patch16_224()
cp = torch.load("../vit_s_k710_dl_from_giant.pth", map_location="cpu")

for model_key in ["model", "module"]:
    if model_key in cp:
        cp = cp[model_key]
        break

cp_model.load_state_dict(cp)
cp_model.eval()
cp_model.cuda()


# Model
class HNeRVMAE(nn.Module):
    def __init__(self, embedding=None):
        super().__init__()
        self.embedding = embedding

        self.conv1 = nn.Conv2d(
            192,
            96 * 4**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px1 = nn.PixelShuffle(4)

        self.conv2 = nn.Conv2d(
            96,
            48 * 2**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px2 = nn.PixelShuffle(2)

        self.conv3 = nn.Conv2d(
            48,
            24 * 2**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px3 = nn.PixelShuffle(2)

        self.conv4 = nn.Conv2d(
            24, 3, kernel_size=3, stride=(1, 1), padding=ceil((3 - 1) // 2)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.px1(self.conv1(x)))
        x = self.act(self.px2(self.conv2(x)))
        x = self.act(self.px3(self.conv3(x)))
        x = self.act(self.conv4(x))

        x = x.permute(0, 2, 3, 1) * 255
        return x


model = HNeRVMAE()

optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

# Training

tqdm_batch = tqdm(
    iterable=beauty_dataloader,
    desc="Epoch {}".format(0),
    total=len(beauty_dataloader),
    unit="it",
)

for e in range(10):
    for batch_idx, data in enumerate(tqdm_batch):
        data = data.permute(3, 0, 1, 2).unsqueeze(0).cuda()

        # Forward feature ckpt
        feature = cp_model.forward_features(data)
        input = feature.reshape(12, 192, 14, 14)

        optimizer.zero_grad()

        # HNeRV MAE
        model = HNeRVMAE().cuda()

        output = model(input)

        loss = F.mse_loss(output.reshape(1, 3, 12, 224, 224), data)
        loss.backward()

        optimizer.step()

        tqdm_batch.set_postfix(loss=loss / input.shape[0])
