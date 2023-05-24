import time
import numpy as np
import torch
import torch.nn as nn
from math import ceil


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = (
            kargs["ks"],
            kargs["ngf"],
            kargs["new_ngf"],
            kargs["strd"],
        )
        if kargs["conv_type"] == "pshuffel":
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd != 1 else nn.Identity(),
                nn.Conv2d(
                    ngf * strd**2,
                    new_ngf,
                    ks,
                    1,
                    ceil((ks - 1) // 2),
                    bias=kargs["bias"],
                ),
            )
        elif kargs["conv_type"] == "conv":
            self.downconv = nn.Conv2d(
                ngf, new_ngf, ks + strd, strd, ceil(ks / 2), bias=kargs["bias"]
            )
        elif kargs["conv_type"] == "interpolate":
            self.downconv = nn.Sequential(
                nn.Upsample(
                    scale_factor=1.0 / strd,
                    mode="bilinear",
                ),
                nn.Conv2d(
                    ngf,
                    new_ngf,
                    ks + strd,
                    1,
                    ceil((ks + strd - 1) / 2),
                    bias=kargs["bias"],
                ),
            )

    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = (
            kargs["ks"],
            kargs["ngf"],
            kargs["new_ngf"],
            kargs["strd"],
        )

        if kargs["conv_type"] == "pshuffel":
            self.upconv = nn.Sequential(
                nn.Conv2d(
                    ngf,
                    new_ngf * strd * strd,
                    ks,
                    1,
                    ceil((ks - 1) // 2),
                    bias=kargs["bias"],
                ),
                nn.PixelShuffle(strd) if strd != 1 else nn.Identity(),
            )

        elif kargs["conv_type"] == "conv":
            self.upconv = nn.ConvTranspose2d(
                ngf, new_ngf, ks + strd, strd, ceil(ks / 2)
            )

        elif kargs["conv_type"] == "interpolate":
            self.upconv = nn.Sequential(
                nn.Upsample(
                    scale_factor=strd,
                    mode="bilinear",
                ),
                nn.Conv2d(
                    ngf,
                    new_ngf,
                    strd + ks,
                    1,
                    ceil((ks + strd - 1) / 2),
                    bias=kargs["bias"],
                ),
            )

    def forward(self, x):
        return self.upconv(x)


def NormLayer(norm_type, ch_width):
    if norm_type == "none":
        norm_layer = nn.Identity()
    elif norm_type == "bn":
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == "in":
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


def ActivationLayer(act_type):
    if act_type == "relu":
        act_layer = nn.ReLU(True)
    elif act_type == "leaky":
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == "leaky01":
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == "gelu":
        act_layer = nn.GELU()
    elif act_type == "sin":
        act_layer = Sin
    elif act_type == "swish":
        act_layer = nn.SiLU(inplace=True)
    elif act_type == "softplus":
        act_layer = nn.Softplus()
    elif act_type == "hardswish":
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def OutImg(x, out_bias="tanh"):
    if out_bias == "sigmoid":
        return torch.sigmoid(x)
    elif out_bias == "tanh":
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


######################################################


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs["dec_block"] else DownConv

        self.conv = conv(
            ngf=kargs["ngf"],
            new_ngf=kargs["new_ngf"],
            strd=kargs["strd"],
            ks=kargs["ks"],
            conv_type=kargs["conv_type"],
            bias=kargs["bias"],
        )
        self.norm = NormLayer(kargs["norm"], kargs["new_ngf"])
        self.act = ActivationLayer(kargs["act"])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class HNeRV(nn.Module):
    def __init__(self, args):
        super().__init__()
        ks_dec1, ks_dec2 = [1, 5]

        # BUILD Decoder LAYERS
        decoder_layers = []
        ngf = args.fc_dim  # 192

        self.dec_strds = [4, 2, 2]

        for i, strd in enumerate(self.dec_strds):
            new_ngf = int(round(ngf / 2))

            cur_blk = NeRVBlock(
                dec_block=True,  # upconv
                conv_type=args.conv_type[1],  # convnext pshuffel
                ngf=ngf,
                new_ngf=new_ngf,
                ks=min(ks_dec1 + 2 * i, ks_dec2),
                strd=1,
                bias=True,
                norm=args.norm,
                act=args.act,
            )

            decoder_layers.append(cur_blk)
            ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1)
        self.out_bias = args.out_bias

    def forward(self, input, input_embed=None, encode_only=False):
        if input_embed != None:
            img_embed = input_embed
        else:
            if "pe" in self.embed:
                input = self.pe_embed(input[:, None]).float()
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

write_video(
    "check.mp4",
    output,
    fps=16,
    options={"crf": "10"},
)
