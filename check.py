from yuv import YUVReader
from yuv import ycbcr420_to_444
import torch
import matplotlib.pyplot as plt
import numpy as np


src_reader = YUVReader("../uvg-raw/beauty.yuv", 1920, 1080)
frame = src_reader.read_one_frame(frame_index=100, dst_format="rgb")

# for frame_idx in range(600):
#    rgb = src_reader.read_one_frame(dst_format="rgb")
#    tensor_rgb = torch.from_numpy(rgb).type(torch.FloatTensor)  # unsqueeze(0)

# plt.imshow(np.transpose(tensor_rgb.cpu().numpy(), (1,2,0)))

"""
class VideoDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_interval,
        crop_size=(720, 1080),
        mode="train",
    ):
        self.data_path = data_path
        self.mode = mode
        self.frame_interval = frame_interval
        self.crop_size = crop_size

        self.vr = YUVReader(self.data_path)

        self.data_transform = video_transforms.Compose(
            [video_transforms.CenterCrop(crop_size)]
        )

    def __len__(self):
        return len(self.vr) // self.frame_interval

    def __getitem__(self, index):
        current_idx = index * self.frame_interval
        self.vr.seek(current_idx)
        list_idx = list(range(current_idx, current_idx + self.frame_interval))
        buffer = self.vr.get_batch(list_idx).asnumpy()
        self.vr.seek(0)

        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn(
                    "video {} not correctly loaded".format(
                        self.data_path.split("/")[-1].split(".")[0]
                    )
                )
                index = np.random.randint(self.__len__())
                sample = self.dataset_samples[index]
                buffer = self.load_video(sample)

        buffer = self.data_transform(buffer)

        return buffer.permute(1, 2, 3, 0)  # CTHW to THWC
"""
