from torch.utils.data import DataLoader
from .datasets import VideoDataset
from .yuv import YUVDataset


def build_dataset(name, data_path, frame_interval, crop_size=None):
    if name == "uvghd30":
        dataset = VideoDataset(
            data_path=data_path,
            frame_interval=frame_interval,
            crop_size=crop_size,
        )

    elif name == "uvg-raw":
        dataset = YUVDataset(
            data_path=data_path, frame_interval=frame_interval, crop_size=crop_size
        )

    else:
        raise NotImplementedError("Unsupported Dataset")

    return dataset


def build_dataloader(
    name,
    frame_interval,
    data_path="data/uvghd30/uvghd30.mp4",
    batch_size=5,
    crop_size=224,
):
    if name == "uvghd30":
        dataset = build_dataset(
            name=name,
            data_path=data_path,
            frame_interval=frame_interval,
            crop_size=crop_size,
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    elif name == "uvg-raw":
        dataset = build_dataset(
            name=name,
            data_path=data_path,
            frame_interval=frame_interval,
            crop_size=crop_size,
        )

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=6
        )

    else:
        raise NotImplementedError("Unsupported Dataloader")

    return dataset, dataloader
