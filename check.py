from src.dataset.yuv import YUVReader
from torch.utils.data import Dataset
import numpy as np

src_reader = YUVReader("../uvg-raw/beauty.yuv", 1080, 1920)
frame = src_reader.read_one_frame(frame_index=100, dst_format="rgb")


class YUVDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_interval=4,
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
        buffer = []
        for idx in range(self.frame_interval):
            frame = self.vr.read_one_frame(
                frame_index=index * self.frame_interval + idx, dst_format="rgb"
            )
            buffer.append(frame)

        return np.array(buffer).permute(0, 2, 3, 1)  # TCHW to THWC
