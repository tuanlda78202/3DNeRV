from src.model.nerv3d import NeRV3D
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    model = NeRV3D(
        img_size=(720, 1280),
        frame_interval=4,
        embed_dim=8,
        decode_dim=314,
        embed_size=(9, 16),
        scales=[5, 4, 2, 2],
        lower_width=6,
        reduce=3,
    ).cuda()

    macs, params = get_model_complexity_info(
        model,
        (3, 4, 720, 1280),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))
