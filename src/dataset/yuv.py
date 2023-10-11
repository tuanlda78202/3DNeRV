import os
import torch
import cupy as cp
from cupyx.scipy.ndimage import zoom
from torch.utils.data import Dataset


def ycbcr420_to_rgb(y, uv, order=1):
    """
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)

    return value is 3xhxw RGB float numpy array, in the range of [0, 1]
    """
    YCBCR_WEIGHTS = {"ITU-R_BT.709": (0.2126, 0.7152, 0.0722)}

    uv = zoom(uv, (1, 2, 2), order=order)
    cb = uv[0:1, :, :]
    cr = uv[1:2, :, :]

    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    rgb = cp.concatenate((r, g, b), axis=0)
    rgb = cp.clip(rgb, 0.0, 1.0)

    return rgb


class BaseYUV:
    def __init__(self, src_path, height, width):
        self.src_path = src_path
        self.width = width
        self.height = height
        self.eof = False

    def read_one_frame(self, dst_format="rgb"):
        raise NotImplementedError

    @staticmethod
    def _none_exist_frame(dst_format="rgb"):
        assert dst_format == "rgb"
        return None


class YUVReader(BaseYUV):
    def __init__(self, src_path, height=1080, width=1920, src_format="420"):
        super().__init__(src_path, height, width)

        self.src_path = src_path
        self.src_format = src_format
        self.y_size = width * height
        self.uv_size = width * height // 2
        self.file = open(src_path, "rb")

    def read_one_frame(self, frame_index):
        frame_index += 1
        assert frame_index > 0

        if self.eof:
            return self._none_exist_frame()

        # Seek frame (width * height * 3/2) * frame_index
        frame_offset = self.y_size * 1.5 * (frame_index - 1)
        self.file.seek(int(frame_offset))

        # Read Y and UV component
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)

        if not y or not uv:
            self.eof = True
            return self._none_exist_frame()

        y = cp.frombuffer(y, dtype=cp.uint8).copy().reshape(1, self.height, self.width)
        uv = (
            cp.frombuffer(uv, dtype=cp.uint8)
            .copy()
            .reshape(2, self.height // 2, self.width // 2)
        )

        # Normalize
        y = y.astype(cp.float32) / 255
        uv = uv.astype(cp.float32) / 255

        # Convert YUV to RGB
        rgb = ycbcr420_to_rgb(y, uv)

        return rgb

    def __len__(self):
        one_frame = float(self.y_size * 3 / 2)
        return int(os.path.getsize(self.src_path) / one_frame)

    def close(self):
        self.file.close()


class YUVDataset(Dataset):
    def __init__(
        self,
        data_path,
        frame_interval,
    ):
        self.data_path = data_path
        self.frame_interval = frame_interval

        self.vr = YUVReader(self.data_path)
        self.file = self.vr.file

    def __len__(self):
        return len(self.vr) // self.frame_interval

    def __getitem__(self, index):
        buffer = []

        for idx in range(self.frame_interval):
            frame = self.vr.read_one_frame(
                frame_index=index * self.frame_interval + idx
            )
            buffer.append(frame)  # CHW

        buffer = cp.array(buffer).transpose(0, 2, 3, 1)  # THWC

        return torch.as_tensor(buffer, device="cuda")
