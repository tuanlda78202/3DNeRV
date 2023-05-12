import torch.nn as nn
import torch

input = torch.rand(3, 16, 224, 224)
print(input.shape)
f = nn.Conv3d(
    in_channels=3, out_channels=768, kernel_size=(2, 16, 16), stride=(2, 16, 16)
)
print(f(input).shape)
