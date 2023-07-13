from src.dataset.yuv import YUVReader
import numpy as np
from torch.utils.data import Dataset


src_reader = YUVReader("../uvg-raw/beauty.yuv", 1080, 1920)
frame = src_reader.read_one_frame(frame_index=100, dst_format="rgb")


class YUVDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_interval,
        crop_size=(1080, 1920),
        mode="train",
    ):
        self.data_path = data_path
        self.mode = mode
        self.frame_interval = frame_interval
        self.crop_size = crop_size

        self.vr = YUVReader(self.data_path, *crop_size)
        self.file = self.vr.file

    def __len__(self):
        return len(self.vr) // self.frame_interval

    def __getitem__(self, index):
        current_idx = index * self.frame_interval
        self.vr.seek(current_idx)
        list_idx = list(range(current_idx, current_idx + self.frame_interval))
        # buffer = self.vr.get_batch(list_idx).asnumpy()
        self.vr.seek(0)

        buffer = self.data_transform(buffer)

        return buffer.permute(1, 2, 3, 0)  # CTHW to THWC


data = YUVDataset("../uvg-raw/beauty.yuv", frame_interval=4)

print(len(data))
