import wandb
import torch
from src.model.nerv3d import NeRV3D
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    model_3m = NeRV3D(
        img_size=(1080, 1920),
        frame_interval=4,
        embed_dim=8,
        decode_dim=305,
        embed_size=(9, 16),
        scales=[5, 4, 3, 2],
        lower_width=6,
        reduce=3,
    ).cuda()

    macs_3m, params_3m = get_model_complexity_info(
        model_3m,
        (3, 4, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 3M model: ", macs_3m))
    print("{:<30}  {:<8}".format("Number of parameters 3M model: ", params_3m))

    ###############################################################################
    model_6m = NeRV3D(
        img_size=(1080, 1920),
        frame_interval=4,
        embed_dim=8,
        decode_dim=440,
        embed_size=(9, 16),
        scales=[5, 4, 3, 2],
        lower_width=6,
        reduce=3,
    ).cuda()

    macs_6m, params_6m = get_model_complexity_info(
        model_6m,
        (3, 4, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 6M model: ", macs_6m))
    print("{:<30}  {:<8}".format("Number of parameters 6M model: ", params_6m))

    ###############################################################################
    model_12m = NeRV3D(
        img_size=(1080, 1920),
        frame_interval=4,
        embed_dim=8,
        decode_dim=634,
        embed_size=(9, 16),
        scales=[5, 4, 3, 2],
        lower_width=6,
        reduce=3,
    ).cuda()

    macs_12m, params_12m = get_model_complexity_info(
        model_12m,
        (3, 4, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 12M model: ", macs_12m))
    print("{:<30}  {:<8}".format("Number of parameters 12M model: ", params_12m))
