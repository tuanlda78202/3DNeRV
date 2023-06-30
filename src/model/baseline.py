from src.backbone.videomaev2 import vit_small_patch16_224
import torch.nn as nn
from math import ceil
import torch
import torch.nn.functional as F


class NeRVBlock2D(nn.Module):
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


class NeRVBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, scale, ks=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch * scale**2,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.ps = nn.PixelShuffle(scale)
        self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.ps(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.act(self.norm(x))

        return x


class HNeRVMae(nn.Module):
    def __init__(self, bs, fi=None, c3d=False):
        super().__init__()

        self.bs = bs
        self.fi = fi
        self.c3d = c3d
        self.encoder = self.vmae_pretrained()

        # 2D
        self.blk1 = NeRVBlock2D(in_ch=192, out_ch=96, scale=2)
        self.blk2 = NeRVBlock2D(in_ch=96, out_ch=48, scale=2)
        self.blk3 = NeRVBlock2D(in_ch=48, out_ch=24, scale=2)
        self.blk4 = NeRVBlock2D(in_ch=24, out_ch=12, scale=2)
        self.final2d = nn.Conv2d(
            12, 3, kernel_size=3, stride=(1, 1), padding=ceil((3 - 1) // 2)
        )
        # 3D

        self.blk3d_1 = NeRVBlock3D(in_ch=192, out_ch=360, scale=2)
        self.blk3d_2 = NeRVBlock3D(in_ch=360, out_ch=80, scale=2)
        self.blk3d_3 = NeRVBlock3D(in_ch=80, out_ch=18, scale=2)
        self.blk3d_4 = NeRVBlock3D(in_ch=18, out_ch=3, scale=2)

        self.final3d = nn.Conv3d(6, 3, kernel_size=3, stride=1, padding=1)
        self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        x = self.encoder.forward_features(x)

        B, N, D = x.shape
        dim_encoder = int(N * D / 40**2 / self.fi)  # 192
        x = x.reshape(self.bs, dim_encoder, self.fi, 40, 40)

        if self.c3d == True:
            x = self.blk3d_1(x)
            x = self.blk3d_2(x)
            x = self.blk3d_3(x)
            x = self.blk3d_4(x)
            # x = self.act(self.norm(self.final3d(x)))

            return x.permute(0, 2, 1, 3, 4)

        else:
            x = self.blk1(x)
            x = self.blk2(x)
            x = self.blk3(x)
            x = self.blk4(x)
            x = self.act(self.norm(self.final2d(x)))

            return x.permute(0, 2, 3, 1)

    def vmae_pretrained(
        self,
        vmae_cp="/home/tuanlda78202/ckpt/vit_s_k710_dl_from_giant.pth",
    ):
        pretrained_mae = vit_small_patch16_224(
            all_frames=self.fi,
        )  # patch_size=32, embed_dim=192

        checkpoint = torch.load(vmae_cp, map_location="cpu")

        for model_key in ["model", "module"]:
            if model_key in checkpoint:
                checkpoint = checkpoint[model_key]
                break

        pretrained_mae.load_state_dict(checkpoint)
        pretrained_mae.eval()
        pretrained_mae.cuda()

        for param in pretrained_mae.parameters():
            param.requires_grad = False

        return pretrained_mae


class HNeRVMaeDecoder(nn.Module):
    def __init__(self, fi=None, bias=True):
        super().__init__()

        self.fi = fi
        self.bias = bias

        self.blk3d_1 = NeRVBlock3D(in_ch=192, out_ch=360, scale=2, bias=self.bias)
        self.blk3d_2 = NeRVBlock3D(in_ch=360, out_ch=80, scale=2, bias=self.bias)
        self.blk3d_3 = NeRVBlock3D(in_ch=80, out_ch=18, scale=2, bias=self.bias)
        self.blk3d_4 = NeRVBlock3D(in_ch=18, out_ch=3, scale=2, bias=self.bias)

    def forward(self, x):
        B, N, D = x.shape
        dim_encoder = int(N * D / 40**2 / self.fi)
        x = x.reshape(B, dim_encoder, self.fi, 40, 40)

        x = self.blk3d_1(x)
        x = self.blk3d_2(x)
        x = self.blk3d_3(x)
        x = self.blk3d_4(x)
        return x.permute(0, 2, 1, 3, 4)
