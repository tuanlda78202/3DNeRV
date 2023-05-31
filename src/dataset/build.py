from torch.utils.data import DataLoader
from .datasets import VideoDataset


def build_dataset(name, frame_interval=12):
    if name == "uvghd30":
        dataset = VideoDataset(
            data_path="data/uvghd30/uvghd30.mp4",
            mode="train",
            frame_interval=12,
            crop_size=224,
            short_side_size=256,
        )

    else:
        raise NotImplementedError("Unsupported Dataset")

    return dataset


def build_dataloader(name, batch_size=5):
    if name == "uvghd30":
        dataset = build_dataset(name="uvghd30", frame_interval=12)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    else:
        raise NotImplementedError("Unsupported Dataloader")

    return dataloader
