from yuv import YUVReader
from yuv import ycbcr420_to_444
import torch
import matplotlib.pyplot as plt
import numpy as np


src_reader = YUVReader("../uvg-raw/beauty.yuv", 1920, 1080)

for frame_idx in range(600):
    rgb = src_reader.read_one_frame(dst_format="rgb")
    tensor_rgb = torch.from_numpy(rgb).type(torch.FloatTensor)  # unsqueeze(0)

# plt.imshow(np.transpose(tensor_rgb.cpu().numpy(), (1,2,0)))
