from all.model.hnerv import HNeRV, HNeRVDecoder
from all.backbone.videomae import vit_base_patch16_224
import torch


# Embedding 3D output of VideoMAE fed to Decoder HNerV (config to 3D kernel)
class VideomaeHnerv:
    def __init__(self) -> None:
        pass
