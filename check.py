from src.dataset.datasets import VideoDataset
from src.backbone.videomaev2 import vit_small_patch16_224
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.nn as nn
from math import ceil
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

from src.dataset.loader import get_video_loader

video_loader = get_video_loader()

vr = video_loader("data/uvghd30/uvghd30.mp4")

print(len(vr[0]))
