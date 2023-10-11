from src.dataset.yuv import YUVDataset
import wandb
import torch
from torch.utils.data import DataLoader

wandb.init(project="nerv3d", entity="tuanlda78202", mode="offline")

dataset = YUVDataset("../uvg/beauty.yuv", frame_interval=2)

dataloader = DataLoader(dataset, batch_size=2)


for batch_idx, data in enumerate(dataloader):
    # BTHWC to BCTHW
    data = data.permute(0, 4, 1, 2, 3).cuda()

    # BCTHW to BTCHW
    data = data.permute(0, 2, 1, 3, 4)

    valid_data = torch.mul(data, 255).type(torch.uint8)
    valid_data = valid_data.squeeze(0).cpu().detach().numpy()

    wandb.log(
        {
            "data": wandb.Video(valid_data, fps=2, format="mp4"),
        }
    )

"""
for x in range(10):
    data = dataset[x].unsqueeze(0).permute(0, 4, 1, 2, 3).cuda()
    data = data.permute(0, 2, 1, 3, 4)
    data = torch.mul(data, 255).type(torch.uint8)
    data = data.squeeze(0).cpu().detach().numpy()

    wandb.log(
        {
            "data": wandb.Video(data, fps=2, format="mp4"),
        }
    )
"""
