from src.model.hnerv3d import HNeRVMae
import torch
from ptflops import get_model_complexity_info
from torchsummary import summary

"""
with torch.cuda.device(0):
    model = HNeRVMae(
        img_size=(720, 1280),
        frame_interval=4,
        embed_dim=8,
        decode_dim=314,
        embed_size=(9, 16),
        scales=[5, 4, 2, 2],
        lower_width=6,
        reduce=3,
        ckpt_path="../ckpt/vit_s_k710_dl_from_giant.pth",
    ).cuda()

    model = torch.compile(model)

    macs, params = get_model_complexity_info(
        model,
        (3, 4, 720, 1280),
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True,
    )

    print("{:<30}  {:<8}".format("Computational complexity: ", macs))
    print("{:<30}  {:<8}".format("Number of parameters: ", params))

    print(summary(model, (3, 4, 720, 1280), batch_size=1, device="cuda"))
"""
