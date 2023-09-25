import sys
import os

sys.path.append(os.getcwd())

from src.model.nerv3d import NeRV3D
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary

with torch.cuda.device(0):
    model_12m = NeRV3D(
        img_size=(1080, 1920),
        frame_interval=2,
        embed_dim=8,
        decode_dim=634,
        embed_size=(9, 16),
        scales=[5, 4, 3, 2],
        lower_kernel=1,
        upper_kernel=5,
        lower_width=6,
        reduce=3,
    ).cuda()

    macs_12m, params_12m = get_model_complexity_info(
        model_12m,
        (3, 2, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 12M model: ", macs_12m))
    print("{:<30}  {:<8}".format("Number of parameters 12M model: ", params_12m))

    print(summary(model_12m, (3, 2, 1080, 1920)))
