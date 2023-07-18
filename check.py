<<<<<<< HEAD
import wandb
import os
import torch
from src.dataset.yuv import YUVDataset
import random
from src.model.hnerv3d import HNeRVMae

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"


dataset = YUVDataset(
    data_path="../uvg-raw/beauty.yuv", frame_interval=10, crop_size=(720, 1280)
)

index_random = random.randint(0, len(dataset) - 1)
data = dataset[index_random].permute(3, 0, 1, 2).unsqueeze(0).cuda()

model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=10,
    embed_size=(18, 32),
    scales=[5, 2, 2, 2],
).cuda()

output = model(data)

# BTCHW
data = torch.mul(data, 255).permute(0, 2, 1, 3, 4).type(torch.uint8)
output = torch.mul(output, 255).permute(0, 2, 1, 3, 4).type(torch.uint8)

output = output.cpu().detach().numpy()
data = data.cpu().detach().numpy()

wandb.init(
    project="vmae-nerv3d-1ke",
    name="beauty-raw720p-400e",
)

wandb.log(
    {
        "pred": wandb.Video(output, fps=4, format="mp4"),
        "gt": wandb.Video(data, fps=4, format="mp4"),
    },
)
=======
# from src.dataset.yuv import YUVDataset
from src.model.hnerv3d import HNeRVMae
from torchsummary import summary

model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=4,
    embed_dim=8,
    decode_dim=661,
    embed_size=(9, 16),
    scales=[5, 4, 2, 2],
    lower_width=6,
    reduce=3,
)

print(summary(model, (3, 4, 720, 1280), batch_size=1))
>>>>>>> a8e984d8c70de51e6c324f97931e6cd584128962
