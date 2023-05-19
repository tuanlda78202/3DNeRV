import os
import decord
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.utils.data import Dataset

decord.bridge.set_bridge("torch")


class VideoDataSet(Dataset):
    def __init__(self, config):
        if os.path.isfile(config.data_path):
            self.video = decord.VideoReader(config.data_path)

        else:
            self.video = [
                os.path.join(config.data_path, x)
                for x in sorted(os.listdir(config.data_path))
            ]

        # Resize the input video and center crop
        self.crop_list, self.resize_list = config.crop_list, config.resize_list

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
        if self.crop_list != "-1":
            crop_h, crop_w = [int(x) for x in self.crop_list.split("_")[:2]]
            if "last" not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))

        if self.resize_list != "-1":
            if "_" in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split("_")]
                img = interpolate(img, (resize_h, resize_w), "bicubic")
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw, "bicubic")

        if "last" in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))

        return img
