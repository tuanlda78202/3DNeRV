import time
import numpy as np
import torch
import torch.nn as nn
from math import ceil
import torch
from src.backbone.videomaev2 import vit_small_patch16_224
from src.data.vmae_feat import *

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401


class UpConv(nn.Module):
    """Conv2D + PixelShuffle"""

    def __init__(self, **kargs):
        super(UpConv, self).__init__()

        kernel_size, in_ch, out_ch, scale = (
            kargs["kernel_size"],
            kargs["in_ch"],
            kargs["out_ch"],
            kargs["scale"],
        )

        self.upconv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch * scale * scale,
                kernel_size=kernel_size,
                stride=1,
                padding=ceil((kernel_size - 1) // 2),
                bias=kargs["bias"],
            ),
            nn.PixelShuffle(scale) if scale != 1 else nn.Identity(),
        )

    def forward(self, x):
        return self.upconv(x)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = UpConv(
            in_ch=kargs["in_ch"],
            out_ch=kargs["out_ch"],
            scale=kargs["scale"],
            kernel_size=kargs["kernel_size"],
            bias=kargs["bias"],
        )

        self.norm = nn.Identity()

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Exp1(nn.Module):
    """VideoMAE 3D Embedding to HNeRV"""

    def __init__(self, embedding, encoder):
        super().__init__()

        self.encoder = vit_small_patch16_224()

        self.decoder_layers = []

        in_ch = 192
        self.dec_scale = [4, 2, 2]

        for i, scale in enumerate(self.dec_scale):
            out_ch = int(round(in_ch / 2))

            block = NeRVBlock(
                in_ch=in_ch, out_ch=out_ch, kernel_size=3, strd=1, bias=True
            )

            self.decoder_layers.append(block)

            in_ch = out_ch

        self.decoder = nn.ModuleList(self.decoder_layers)
        self.head_layer = nn.Conv2d(in_ch, 3, 3, 1, 1)

    def forward(self, input):
        img_embed = self.encoder(input)

        embed_list = [img_embed]

        # Decoder
        dec_start = time.time()

        output = self.decoder[0](img_embed)

        n, c, h, w = output.shape
        output = (
            output.view(n, -1, self.fc_h, self.fc_w, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, -1, self.fc_h * h, self.fc_w * w)
        )

        embed_list.append(output)

        for layer in self.decoder[1:]:
            output = layer(output)
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dec_time = time.time() - dec_start

        return img_out, embed_list, dec_time


class HNeRVMAE(nn.Module):
    def __init__(self, embedding=None):
        super().__init__()
        self.embedding = embedding

        self.conv1 = nn.Conv2d(
            192,
            96 * 4**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px1 = nn.PixelShuffle(4)

        self.conv2 = nn.Conv2d(
            96,
            48 * 2**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px2 = nn.PixelShuffle(2)

        self.conv3 = nn.Conv2d(
            48,
            24 * 2**2,
            kernel_size=3,
            stride=(1, 1),
            padding=ceil((3 - 1) // 2),
        )

        self.px3 = nn.PixelShuffle(2)

        self.conv4 = nn.Conv2d(
            24, 3, kernel_size=3, stride=(1, 1), padding=ceil((3 - 1) // 2)
        )

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.px1(self.conv1(x)))
        x = self.act(self.px2(self.conv2(x)))
        x = self.act(self.px3(self.conv3(x)))
        x = self.act(self.conv4(x))

        x = x.permute(0, 2, 3, 1) * 255
        return x


import numpy as np
from torchvision.io import write_video

arrays = np.load("/home/tuanlda78202/3ai24/vit_small_patch16_224/beauty.npy")
frame = torch.from_numpy(arrays)[0]
input = frame.reshape(16, 192, 14, 14)

model = HNeRVMAE()
output = model(input)

print(output.shape)
"""
write_video(
    "check.mp4",
    output,
    fps=16,
    options={"crf": "10"},
)
"""
