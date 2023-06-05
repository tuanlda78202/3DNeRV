from src.backbone.videomaev2 import vit_small_patch16_224
import torch.nn as nn
from math import ceil
import torch
import torch.nn.functional as F


class NeRVBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, scale, ks=3, stride=1, padding=ceil((3 - 1) // 2)
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch * scale**2,
            kernel_size=ks,
            stride=stride,
            padding=padding,
        )
        self.ps = nn.PixelShuffle(scale)
        self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.ps(self.conv(x))))


class HNeRVMae(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = self.vmae_pretrained()

        self.blk1 = NeRVBlock(in_ch=192, out_ch=96, scale=4)
        self.blk2 = NeRVBlock(in_ch=96, out_ch=48, scale=2)
        self.blk3 = NeRVBlock(in_ch=48, out_ch=24, scale=2)

        self.final = nn.Conv2d(
            24, 3, kernel_size=3, stride=(1, 1), padding=ceil((3 - 1) // 2)
        )

    def forward(self, x):
        # Encoder
        x = self.encoder.forward_features(x)
        x = x.reshape(12, 192, 14, 14)

        # Decoder
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = F.gelu(self.final(x))

        return x.permute(0, 2, 3, 1)

    def vmae_pretrained(
        self,
        vmae_model=vit_small_patch16_224(),
        vmae_cp="../vit_s_k710_dl_from_giant.pth",
    ):
        pretrained_mae = vmae_model
        checkpoint = torch.load(vmae_cp, map_location="cpu")

        for model_key in ["model", "module"]:
            if model_key in checkpoint:
                checkpoint = checkpoint[model_key]
                break

        pretrained_mae.load_state_dict(checkpoint)
        pretrained_mae.eval()
        pretrained_mae.cuda()

        return pretrained_mae
