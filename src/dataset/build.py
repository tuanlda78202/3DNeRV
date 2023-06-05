from torch.utils.data import DataLoader
from .datasets import VideoDataset


def build_dataset(name, data_path, frame_interval, crop_size):
    if name == "uvghd30":
        dataset = VideoDataset(
            data_path=data_path,
            mode="train",
            frame_interval=frame_interval,
            crop_size=crop_size,
            short_side_size=256,
        )

    else:
        raise NotImplementedError("Unsupported Dataset")

    return dataset


def build_dataloader(
    name,
    data_path="data/uvghd30/uvghd30.mp4",
    batch_size=5,
    frame_interval=12,
    crop_size=224,
):
    if name == "uvghd30":
        dataset = build_dataset(
            name=name,
            data_path=data_path,
            frame_interval=frame_interval,
            crop_size=crop_size,
        )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    else:
        raise NotImplementedError("Unsupported Dataloader")

    return dataloader
