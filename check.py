from src.dataset.build import build_dataloader
from tqdm import tqdm

dataloader = build_dataloader(name="uvghd30", batch_size=5)

for ep in range(10):
    tqdm_batch = tqdm(
        iterable=dataloader,
        desc="Epoch {}".format(ep),
        total=len(dataloader),
        unit="it",
    )
    for batch_idx, data in enumerate(tqdm_batch):
        print(data.shape)
