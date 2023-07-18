# from src.dataset.yuv import YUVDataset
from src.model.hnerv3d import HNeRVMae
from torchsummary import summary

model = HNeRVMae(
    img_size=(720, 1280),
    frame_interval=4,
    embed_dim=8,
    decode_dim=661,
    embed_size=(9, 16),
    scales=[5, 4, 2, 2],
    lower_width=6,
    reduce=3,
)

print(summary(model, (3, 4, 720, 1280), batch_size=1))
