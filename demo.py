from src.dataset.build import build_dataloader
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from src.model.baseline import HNeRVMae

# DataLoader
BATCH_SIZE = 1
dataloader = build_dataloader(name="uvghd30", batch_size=BATCH_SIZE)

# Model
model = HNeRVMae()

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

        optimizer.zero_grad()

        # HNeRV MAE
        model = HNeRVMae().cuda()

        output = model(data)

        loss = F.mse_loss(output.reshape(1, 3, 12, 224, 224), data)
        loss.backward()

        optimizer.step()

        tqdm_batch.set_postfix(loss=loss)
