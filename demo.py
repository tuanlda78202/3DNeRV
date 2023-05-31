from src.dataset.build import build_dataloader
from src.backbone.videomaev2 import vit_small_patch16_224
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from model import HNeRVMAE

# ENV
BATCH_SIZE = 1

# DataLoader
dataloader = build_dataloader(name="uvghd30", batch_size=BATCH_SIZE)

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
model = HNeRVMAE()

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
