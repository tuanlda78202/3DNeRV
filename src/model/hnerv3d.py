from ..backbone.videomaev2 import vit_small_patch16_224
import torch
from torch import nn
import torch.functional as F
from math import ceil, sqrt
from typing import List, Tuple, Union
import numpy as np

def vmae_pretrained(
    ckt_path=None, 
    model_fn=vit_small_patch16_224,
    **kwargs,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vmae: nn.Module = model_fn(**kwargs)
    
    if ckt_path is not None:
        ckt = torch.load(ckt_path, map_location="cpu")
        for model_key in ["model", "module"]:
            if model_key in ckt:
                ckt = ckt[model_key]
                break
        vmae.load_state_dict(ckt)
    
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
        norm_fn = nn.Identity,
        act_fn = nn.GELU,
    ):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels,
            out_channels * scale ** 2,
            kernel_size,
            stride=1,
            padding=ceil((kernel_size-1) // 2),
            bias=bias
        )
        self.ps = nn.PixelShuffle(scale)
        self.norm = norm_fn()
        self.act = act_fn()

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.ps(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.act(self.norm(x))
        return x
    
class HNeRVMae(nn.Module):
    def __init__(
        self,
        img_size: Tuple = (720, 1280), # shape-impact
        frame_interval: int = 4, # shape-impact
        
        encode_length: int = 6300,
        lower_kernel: int = 1,
        upper_kernel: int = 5,
        scales: List = [5, 2, 2], # shape-impact
        reduce: float = 3,
        lower_width: int = 12,
        embed_size: Tuple = (36, 64), # shape-impact
        bias: bool = True,
        
        norm_fn=nn.Identity,
        act_fn=nn.GELU,
        out_fn=nn.Sigmoid,

        model_fn=vit_small_patch16_224,
        ckt_path=None,
    ):
        super().__init__()

        self.encoder = vmae_pretrained(
            ckt_path, model_fn, all_frames=frame_interval, img_size=img_size)
        hidden_dim = self.encoder.embed_dim
        num_patches = self.encoder.patch_embed.num_patches

        assert encode_length <= num_patches
        self.intermediate = nn.AdaptiveAvgPool1d(encode_length)
        
        self.embed_h, self.embed_w = embed_size
        assert (hidden_dim * encode_length) % (self.embed_h * self.embed_w * frame_interval) == 0
        assert self.embed_h * np.prod(scales) == img_size[0] and self.embed_w * np.prod(scales) == img_size[1]
        self.embed_dim = (hidden_dim * encode_length) // (self.embed_h * self.embed_w * frame_interval)
        
        self.frame_interval = frame_interval

        self.decoder = []
        ngf = self.embed_dim
        for i, scale in enumerate(scales):
            reduction = sqrt(scale) if reduce==-1 else reduce
            new_ngf = int(max(round(ngf / reduction), lower_width))
            upsample_blk = NeRVBlock3D(
                scale, ngf, new_ngf, kernel_size=min(lower_kernel + 2*i, upper_kernel), 
                bias=bias, norm_fn=norm_fn, act_fn=act_fn
            )
            self.decoder.append(upsample_blk)
            ngf = new_ngf
        
        self.decoder = nn.Sequential(*self.decoder)
        self.head_proj = nn.Conv3d(ngf, 3, 3, 1, 1)
        self.head_norm = norm_fn()
        self.out = out_fn()

    def forward(self, x: torch.Tensor):
        x = self.encoder.forward_features(x)
        x = self.intermediate(x.permute(0, 2, 1)).permute(0, 2, 1)

        B, _, _ = x.shape
        x = x.reshape(B, self.embed_dim, self.frame_interval, self.embed_h, self.embed_w)
        
        x = self.decoder(x)
        x = self.out(self.head_norm(self.head_proj(x)))
        
        return x.permute(0, 2, 1, 3, 4)

class HNeRVMaeDecoder(nn.Module):
    def __init__(
        self,
        model: HNeRVMae  
    ):
        super().__init__()
        self.decoder = model.decoder
        self.head_proj = model.head_proj
        self.head_norm = model.head_norm
        self.out = model.out

        self.embed_dim = model.embed_dim
        self.frame_interval = model.frame_interval
        self.embed_h, self.embed_w = model.embed_h, model.embed_w
    
    def forward(self, embedding: torch.Tensor):
        B, _, _ = embedding.shape
        embedding = embedding.reshape(B, self.embed_dim, self.frame_interval, self.embed_h, self.embed_w)

        embedding = self.decoder(embedding)
        embedding = self.head_norm(self.head_proj(embedding))
        
        return self.out(embedding)


            

