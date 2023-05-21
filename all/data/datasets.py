import os
import sys

sys.path.append(os.getcwd())
import numpy as np
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.utils.data import Dataset
from decord import VideoReader, bridge, cpu
from utils.util import load_yaml

bridge.set_bridge("torch")


class VideoDataSet(Dataset):
    def __init__(self, config):
        self.config_data = config["dataloader"]
        self.data_path = self.config_data["data_path"]
        self.crop_list = self.config_data["crop_list"]
        self.resize_list = self.config_data["resize_list"]

        if os.path.isfile(self.data_path):
            self.video = VideoReader(self.data_path, num_threads=1, ctx=cpu(0))

        else:
            self.video = [
                os.path.join(self.data_path, x)
                for x in sorted(os.listdir(self.data_path))
            ]

        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))

        norm_idx = float(idx) / len(self.video)
        sample = {"img": tensor_image, "idx": idx, "norm_idx": norm_idx}

        return sample

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1, 0, 1)

        return img / 255.0

    def img_transform(self, img):
        if self.crop_list != -1:
            crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]

            if "last" not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))

        if self.resize_list != -1:
            if "_" in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                img = interpolate(img, (resize_h, resize_w), "bicubic")

            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, "bicubic")

        if "last" in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))

        return img

    @staticmethod
    def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
        # Clip video with random start > end
        converted_len = int(clip_len * frame_sample_rate)
        end_idx = np.random.randint(converted_len, seg_len)
        str_idx = end_idx - converted_len

        index = np.linspace(str_idx, end_idx, num=clip_len)
        index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)

        return index

    def get_batch(self, clip_len, frame_sample_rate):
        self.video.seek(0)
        index = self.sample_frame_indices(
            clip_len, frame_sample_rate, seg_len=len(self.video)
        )
        buffer = self.video.get_batch(index).numpy()

        return buffer
