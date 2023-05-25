from src.dataset.datasets import VideoDataset

dataset = VideoDataset(
    anno_path="/home/tuanlda78202/3ai24/test.csv",
    data_root="/home/tuanlda78202/3ai24/uvg",
    mode="test",
)

print(dataset[0])
