from torch.utils.data import DataLoader
from .yuv import YUVDataset


def build_data(
    frame_interval,
    data_path,
    num_workers=1,
    batch_size=1,
):
    dataset = YUVDataset(data_path=data_path, frame_interval=frame_interval)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataset, dataloader
