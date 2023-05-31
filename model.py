import torch
import torch.nn as nn
from math import ceil


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
