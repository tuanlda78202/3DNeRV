import os
import sys
import torch

sys.path.append(os.getcwd())
from src.model.nerv3d import NeRV3D
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
    # Embedding: 300 * 9 * 16 * 32 = 1.38M
    ###############################################################################
    # 1325.72 GMac & 1.67M (3.05M)
    model_3m = NeRV3D(
        arch_mode="test",
        img_size=(1080, 1920),
        frame_interval=2,
        embed_dim=32,
        decode_dim=52,
        embed_size=(9, 16),
        scales=[5, 3, 2, 2, 2],
        lower_width=12,
        lower_kernel=1,
        upper_kernel=5,
        reduce=1.2,
    ).cuda()

    macs_3m, params_3m = get_model_complexity_info(
        model_3m,
        (3, 2, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 3M model: ", macs_3m))
    print("{:<30}  {:<8}".format("Number of parameters 3M model: ", params_3m))

    ###############################################################################
    # 2126.84 GMac & 4.68M (6.06M)
    model_6m = NeRV3D(
        arch_mode="test",
        img_size=(1080, 1920),
        frame_interval=2,
        embed_dim=32,
        decode_dim=88,
        embed_size=(9, 16),
        scales=[5, 3, 2, 2, 2],
        lower_width=12,
        lower_kernel=1,
        upper_kernel=5,
        reduce=1.2,
    ).cuda()

    macs_6m, params_6m = get_model_complexity_info(
        model_6m,
        (3, 2, 1080, 1920),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity 6M model: ", macs_6m))
    print("{:<30}  {:<8}".format("Number of parameters 6M model: ", params_6m))

    ###############################################################################
    # 3700.0 GMac & 10.62M (12M)
    model_12m = NeRV3D(
        arch_mode="test",
        img_size=(1080, 1920),
        frame_interval=2,
        embed_dim=32,
        decode_dim=132,
        embed_size=(9, 16),
        scales=[5, 3, 2, 2, 2],
        lower_width=12,
        lower_kernel=1,
        upper_kernel=5,
        reduce=1.2,
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
