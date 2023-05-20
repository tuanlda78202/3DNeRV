########## HNeRV
import torch
import torch.nn as nn
from math import ceil


def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2**quant_bit

    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed

    return img_embed


def OutImg(x, out_bias="tanh"):
    if out_bias == "sigmoid":
        return torch.sigmoid(x)

    elif out_bias == "tanh":
        return (torch.tanh(x) * 0.5) + 0.5

    else:
        return x + float(out_bias)


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


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


###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################


class PositionEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        if "pe" in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split("_")[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if "pe" in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


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


class ModConv(nn.Module):
    def __init__(self, **kargs):
        super(ModConv, self).__init__()
        mod_ks, mod_groups, ngf = kargs["mod_ks"], kargs["mod_groups"], kargs["ngf"]
        self.mod_conv_multi = nn.Conv2d(
            ngf,
            ngf,
            mod_ks,
            1,
            (mod_ks - 1) // 2,
            groups=(ngf if mod_groups == -1 else mod_groups),
        )
        self.mod_conv_sum = nn.Conv2d(
            ngf,
            ngf,
            mod_ks,
            1,
            (mod_ks - 1) // 2,
            groups=(ngf if mod_groups == -1 else mod_groups),
        )

    def forward(self, x):
        sum_att = self.mod_conv_sum(x)
        multi_att = self.mod_conv_multi(x)
        return torch.sigmoid(multi_att) * x + sum_att
