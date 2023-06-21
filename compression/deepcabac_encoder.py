# DeepCABAC compression module
import deepCABAC
from torch.optim import Adam
from src.model.baseline import HNeRVMae
from src.evaluation.evaluation import resume_checkpoint
import time
from tqdm import tqdm
import numpy as np

BATCH_SIZE = 5
FRAME_INTERVAL = 6
CROP_SIZE = 640

model = HNeRVMae(bs=BATCH_SIZE, fi=FRAME_INTERVAL, c3d=True).cuda()
optimizer = Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.99))

start_epoch, model, optimizer = resume_checkpoint(
    model, optimizer, "/home/tuanlda78202/ckpt/beauty-epoch399.pth"
)

model.eval()
model_dict = model.state_dict().copy()

nerv3d_list = [
    "blk3d_1.conv.weight",
    "blk3d_1.conv.bias",
    "blk3d_2.conv.weight",
    "blk3d_2.conv.bias",
    "blk3d_3.conv.weight",
    "blk3d_3.conv.bias",
    "blk3d_4.conv.weight",
    "blk3d_4.conv.bias",
]

model_key = [
    "encoder.patch_embed.proj.weight",
    "encoder.patch_embed.proj.bias",
    "encoder.blocks.0.norm1.weight",
    "encoder.blocks.0.norm1.bias",
    "encoder.blocks.0.attn.q_bias",
    "encoder.blocks.0.attn.v_bias",
    "encoder.blocks.0.attn.qkv.weight",
    "encoder.blocks.0.attn.proj.weight",
    "encoder.blocks.0.attn.proj.bias",
    "encoder.blocks.0.norm2.weight",
    "encoder.blocks.0.norm2.bias",
    "encoder.blocks.0.mlp.fc1.weight",
    "encoder.blocks.0.mlp.fc1.bias",
    "encoder.blocks.0.mlp.fc2.weight",
    "encoder.blocks.0.mlp.fc2.bias",
    "encoder.blocks.1.norm1.weight",
    "encoder.blocks.1.norm1.bias",
    "encoder.blocks.1.attn.q_bias",
    "encoder.blocks.1.attn.v_bias",
    "encoder.blocks.1.attn.qkv.weight",
    "encoder.blocks.1.attn.proj.weight",
    "encoder.blocks.1.attn.proj.bias",
    "encoder.blocks.1.norm2.weight",
    "encoder.blocks.1.norm2.bias",
    "encoder.blocks.1.mlp.fc1.weight",
    "encoder.blocks.1.mlp.fc1.bias",
    "encoder.blocks.1.mlp.fc2.weight",
    "encoder.blocks.1.mlp.fc2.bias",
    "encoder.blocks.2.norm1.weight",
    "encoder.blocks.2.norm1.bias",
    "encoder.blocks.2.attn.q_bias",
    "encoder.blocks.2.attn.v_bias",
    "encoder.blocks.2.attn.qkv.weight",
    "encoder.blocks.2.attn.proj.weight",
    "encoder.blocks.2.attn.proj.bias",
    "encoder.blocks.2.norm2.weight",
    "encoder.blocks.2.norm2.bias",
    "encoder.blocks.2.mlp.fc1.weight",
    "encoder.blocks.2.mlp.fc1.bias",
    "encoder.blocks.2.mlp.fc2.weight",
    "encoder.blocks.2.mlp.fc2.bias",
    "encoder.blocks.3.norm1.weight",
    "encoder.blocks.3.norm1.bias",
    "encoder.blocks.3.attn.q_bias",
    "encoder.blocks.3.attn.v_bias",
    "encoder.blocks.3.attn.qkv.weight",
    "encoder.blocks.3.attn.proj.weight",
    "encoder.blocks.3.attn.proj.bias",
    "encoder.blocks.3.norm2.weight",
    "encoder.blocks.3.norm2.bias",
    "encoder.blocks.3.mlp.fc1.weight",
    "encoder.blocks.3.mlp.fc1.bias",
    "encoder.blocks.3.mlp.fc2.weight",
    "encoder.blocks.3.mlp.fc2.bias",
    "encoder.blocks.4.norm1.weight",
    "encoder.blocks.4.norm1.bias",
    "encoder.blocks.4.attn.q_bias",
    "encoder.blocks.4.attn.v_bias",
    "encoder.blocks.4.attn.qkv.weight",
    "encoder.blocks.4.attn.proj.weight",
    "encoder.blocks.4.attn.proj.bias",
    "encoder.blocks.4.norm2.weight",
    "encoder.blocks.4.norm2.bias",
    "encoder.blocks.4.mlp.fc1.weight",
    "encoder.blocks.4.mlp.fc1.bias",
    "encoder.blocks.4.mlp.fc2.weight",
    "encoder.blocks.4.mlp.fc2.bias",
    "encoder.blocks.5.norm1.weight",
    "encoder.blocks.5.norm1.bias",
    "encoder.blocks.5.attn.q_bias",
    "encoder.blocks.5.attn.v_bias",
    "encoder.blocks.5.attn.qkv.weight",
    "encoder.blocks.5.attn.proj.weight",
    "encoder.blocks.5.attn.proj.bias",
    "encoder.blocks.5.norm2.weight",
    "encoder.blocks.5.norm2.bias",
    "encoder.blocks.5.mlp.fc1.weight",
    "encoder.blocks.5.mlp.fc1.bias",
    "encoder.blocks.5.mlp.fc2.weight",
    "encoder.blocks.5.mlp.fc2.bias",
    "encoder.blocks.6.norm1.weight",
    "encoder.blocks.6.norm1.bias",
    "encoder.blocks.6.attn.q_bias",
    "encoder.blocks.6.attn.v_bias",
    "encoder.blocks.6.attn.qkv.weight",
    "encoder.blocks.6.attn.proj.weight",
    "encoder.blocks.6.attn.proj.bias",
    "encoder.blocks.6.norm2.weight",
    "encoder.blocks.6.norm2.bias",
    "encoder.blocks.6.mlp.fc1.weight",
    "encoder.blocks.6.mlp.fc1.bias",
    "encoder.blocks.6.mlp.fc2.weight",
    "encoder.blocks.6.mlp.fc2.bias",
    "encoder.blocks.7.norm1.weight",
    "encoder.blocks.7.norm1.bias",
    "encoder.blocks.7.attn.q_bias",
    "encoder.blocks.7.attn.v_bias",
    "encoder.blocks.7.attn.qkv.weight",
    "encoder.blocks.7.attn.proj.weight",
    "encoder.blocks.7.attn.proj.bias",
    "encoder.blocks.7.norm2.weight",
    "encoder.blocks.7.norm2.bias",
    "encoder.blocks.7.mlp.fc1.weight",
    "encoder.blocks.7.mlp.fc1.bias",
    "encoder.blocks.7.mlp.fc2.weight",
    "encoder.blocks.7.mlp.fc2.bias",
    "encoder.blocks.8.norm1.weight",
    "encoder.blocks.8.norm1.bias",
    "encoder.blocks.8.attn.q_bias",
    "encoder.blocks.8.attn.v_bias",
    "encoder.blocks.8.attn.qkv.weight",
    "encoder.blocks.8.attn.proj.weight",
    "encoder.blocks.8.attn.proj.bias",
    "encoder.blocks.8.norm2.weight",
    "encoder.blocks.8.norm2.bias",
    "encoder.blocks.8.mlp.fc1.weight",
    "encoder.blocks.8.mlp.fc1.bias",
    "encoder.blocks.8.mlp.fc2.weight",
    "encoder.blocks.8.mlp.fc2.bias",
    "encoder.blocks.9.norm1.weight",
    "encoder.blocks.9.norm1.bias",
    "encoder.blocks.9.attn.q_bias",
    "encoder.blocks.9.attn.v_bias",
    "encoder.blocks.9.attn.qkv.weight",
    "encoder.blocks.9.attn.proj.weight",
    "encoder.blocks.9.attn.proj.bias",
    "encoder.blocks.9.norm2.weight",
    "encoder.blocks.9.norm2.bias",
    "encoder.blocks.9.mlp.fc1.weight",
    "encoder.blocks.9.mlp.fc1.bias",
    "encoder.blocks.9.mlp.fc2.weight",
    "encoder.blocks.9.mlp.fc2.bias",
    "encoder.blocks.10.norm1.weight",
    "encoder.blocks.10.norm1.bias",
    "encoder.blocks.10.attn.q_bias",
    "encoder.blocks.10.attn.v_bias",
    "encoder.blocks.10.attn.qkv.weight",
    "encoder.blocks.10.attn.proj.weight",
    "encoder.blocks.10.attn.proj.bias",
    "encoder.blocks.10.norm2.weight",
    "encoder.blocks.10.norm2.bias",
    "encoder.blocks.10.mlp.fc1.weight",
    "encoder.blocks.10.mlp.fc1.bias",
    "encoder.blocks.10.mlp.fc2.weight",
    "encoder.blocks.10.mlp.fc2.bias",
    "encoder.blocks.11.norm1.weight",
    "encoder.blocks.11.norm1.bias",
    "encoder.blocks.11.attn.q_bias",
    "encoder.blocks.11.attn.v_bias",
    "encoder.blocks.11.attn.qkv.weight",
    "encoder.blocks.11.attn.proj.weight",
    "encoder.blocks.11.attn.proj.bias",
    "encoder.blocks.11.norm2.weight",
    "encoder.blocks.11.norm2.bias",
    "encoder.blocks.11.mlp.fc1.weight",
    "encoder.blocks.11.mlp.fc1.bias",
    "encoder.blocks.11.mlp.fc2.weight",
    "encoder.blocks.11.mlp.fc2.bias",
    "encoder.fc_norm.weight",
    "encoder.fc_norm.bias",
    "encoder.head.weight",
    "encoder.head.bias",
    "blk1.conv.weight",
    "blk1.conv.bias",
    "blk2.conv.weight",
    "blk2.conv.bias",
    "blk3.conv.weight",
    "blk3.conv.bias",
    "blk4.conv.weight",
    "blk4.conv.bias",
    "final2d.weight",
    "final2d.bias",
    "blk3d_1.conv.weight",
    "blk3d_1.conv.bias",
    "blk3d_2.conv.weight",
    "blk3d_2.conv.bias",
    "blk3d_3.conv.weight",
    "blk3d_3.conv.bias",
    "blk3d_4.conv.weight",
    "blk3d_4.conv.bias",
    "final3d.weight",
    "final3d.bias",
]

for key in model_key:
    if key not in nerv3d_list:
        del model_dict[key]

encoder = deepCABAC.Encoder()

interv = 0.1
stepsize = 2 ** (-0.5 * 15)
stepsize_other = 2 ** (-0.5 * 19)
_lambda = 0.0

# Just for model decoder (not included embedding mae)
t0 = time.time()
for name, param in tqdm(model_dict.items()):
    param = param.cpu().numpy()
    if ".weight" in name:
        encoder.encodeWeightsRD(param, interv, stepsize, _lambda)
    else:
        encoder.encodeWeightsRD(param, interv, stepsize_other, _lambda)
print(time.time() - t0)

stream = encoder.finish().tobytes()
print("Compressed size: {:2f} MB".format(1e-6 * len(stream)))

with open("data/beauty.bin", "wb") as f:
    f.write(stream)
