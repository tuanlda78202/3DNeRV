from src.dataset.yuv import YUVDataset

data = YUVDataset("../uvg-raw/beauty.yuv", frame_interval=4, crop_size=(720, 1080))

print(data[0].shape)
