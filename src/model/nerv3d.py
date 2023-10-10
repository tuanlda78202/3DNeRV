import time
import torch
from torch import nn
from math import ceil, sqrt
from typing import List, Tuple
from ..backbone.videomaev2 import vit_small_patch16_224


def vmae_pretrained(
    ckpt_path=None,
    model_fn=vit_small_patch16_224,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmae: nn.Module = model_fn(**kwargs)

    if ckpt_path is not None:
        ckt = torch.load(ckpt_path, map_location="cpu")
        for model_key in ["model", "module"]:
            if model_key in ckt:
                ckt = ckt[model_key]
                break
        vmae.load_state_dict(ckt)

        print("✅Loaded VMAEv2 pretrained weights from {}".format(ckpt_path))

    vmae.eval()
    vmae.to(device)

    for param in vmae.parameters():
        param.requires_grad = False

    return vmae


class NeRVBlock3D(nn.Module):
    def __init__(
        self,
        scale: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        norm_fn=nn.Identity,
        act_fn=nn.GELU,
    ):
        super().__init__()

        self.conv = nn.Conv3d(
            in_channels,
            out_channels * scale**2,
            kernel_size,
            stride=1,
            padding=ceil((kernel_size - 1) // 2),
            bias=bias,
        )
        self.ps = nn.PixelShuffle(scale) if scale != 1 else nn.Identity()
        self.norm = norm_fn()
        self.act = act_fn()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.ps(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.act(self.norm(x))
        return x


class NeRV3D(nn.Module):
    def __init__(
        self,
        frame_interval: int,
        img_size: Tuple = (1080, 1920),
        embed_dim: int = 8,
        embed_size: Tuple = (9, 16),
        decode_dim: int = 305,  # 440 # 634
        lower_kernel: int = 1,
        upper_kernel: int = 5,
        scales: List = [5, 4, 3, 2],
        reduce: float = 3,
        lower_width: int = 6,
        bias: bool = True,
        norm_fn=nn.Identity,
        act_fn=nn.GELU,
        out_fn=nn.Sigmoid,
        model_fn=vit_small_patch16_224,
        ckpt_path=None,
    ):
        super().__init__()

        # Encoder
        self.encoder = vmae_pretrained(
            ckpt_path, model_fn, all_frames=frame_interval, img_size=img_size
        )
        self.hidden_dim = self.encoder.embed_dim
        patch_size = self.encoder.patch_embed.patch_size
        tubelet_size = self.encoder.patch_embed.tubelet_size

        # Embedding
        self.embed_dim = embed_dim
        self.frame_interval = frame_interval
        self.embed_h, self.embed_w = embed_size

        self.hidden_t = frame_interval // tubelet_size
        self.hidden_h, self.hidden_w = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )

        # reduce size of embeddings (hidden to embed)
        self.proj = nn.Sequential(
            nn.Conv3d(
                in_channels=self.hidden_dim,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
            ),
            act_fn(),
            nn.AdaptiveAvgPool3d((self.hidden_t, self.embed_h, self.embed_w)),
        )
        self.embed_dim //= tubelet_size

        # Decoder
        self.decoder = []

        self.decoder.append(
            NeRVBlock3D(
                scale=1,
                in_channels=self.embed_dim,
                out_channels=decode_dim,
                kernel_size=1,
                bias=bias,
                norm_fn=norm_fn,
                act_fn=act_fn,
            )
        )

        ngf = decode_dim
        for i, scale in enumerate(scales):
            reduction = sqrt(scale) if reduce == -1 else reduce
            new_ngf = int(max(round(ngf / reduction), lower_width))

            upsample_blk = NeRVBlock3D(
                scale,
                ngf,
                new_ngf,
                kernel_size=min(lower_kernel + 2 * i, upper_kernel),
                bias=bias,
                norm_fn=norm_fn,
                act_fn=act_fn,
            )
            self.decoder.append(upsample_blk)
            ngf = new_ngf

        self.decoder = nn.Sequential(*self.decoder)
        self.head_proj = nn.Conv3d(ngf, 3, 3, 1, 1)
        self.head_norm = norm_fn()
        self.out = out_fn()

    def forward(self, x: torch.Tensor):
        x = self.encoder.forward_features(x)
        B, _, _ = x.shape
        x = x.reshape(B, self.hidden_dim, self.hidden_t, self.hidden_h, self.hidden_w)

        x = self.proj(x)
        x = x.reshape(
            B, self.embed_dim, self.frame_interval, self.embed_h, self.embed_w
        )  # embedding

        x = self.decoder(x)
        x = self.out(self.head_norm(self.head_proj(x)))

        return x.permute(0, 2, 1, 3, 4)


class NeRV3DEncoder(nn.Module):
    def __init__(self, model: NeRV3D):
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.proj

        self.hidden_dim = model.hidden_dim
        self.hidden_t = model.hidden_t
        self.hidden_h = model.hidden_h
        self.hidden_w = model.hidden_w

    def forward(self, x: torch.Tensor):
        x = self.encoder.forward_features(x)
        B, _, _ = x.shape
        x = x.reshape(B, self.hidden_dim, self.hidden_t, self.hidden_h, self.hidden_w)
        x = self.proj(x)

        return x  # embedding


class NeRV3DDecoder(nn.Module):
    def __init__(self, model: NeRV3D):
        super().__init__()
        self.decoder = model.decoder
        self.head_proj = model.head_proj
        self.head_norm = model.head_norm
        self.out = model.out

        self.embed_dim = model.embed_dim
        self.frame_interval = model.frame_interval
        self.embed_h, self.embed_w = model.embed_h, model.embed_w

    def forward(self, embedding: torch.Tensor):
        B = embedding.shape[0]

        embedding = embedding.reshape(
            B, self.embed_dim, self.frame_interval, self.embed_h, self.embed_w
        )

        dec_start = time.time()

        embedding = self.decoder(embedding)
        embedding = self.head_norm(self.head_proj(embedding))
        output = self.out(embedding)

        torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return output, dec_time
