# 3DNeRV

> "3DNeRV: A Cube-wise Neural Representaion for Videos"<br>
> [Tuan LDA](https://tuanlda78202.github.io)<sup>1, 2, 3</sup>, [Minh Nguyen](https://github.com/minhngt62)<sup>1, 3</sup>, [Thang Nguyen](https://scholar.google.com/citations?user=1NjryzEAAAAJ&hl=en&oi=ao)<sup>3, 4</sup><br>
> <sup>1</sup>Hanoi University of Science and Technology, <sup>2</sup>VinBrain, <sup>3</sup>Viettel High Tech, <sup>4</sup>FPT Software AI Center<br>

- [3DNeRV](#3dnerv)
  - [Abstract](#abstract)
  - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Testing](#testing)
    - [Compression](#compression)
    - [Decoding](#decoding)
  - [Citation](#citation)
  - [Contact](#contact)
  - [Contributors](#contributors)

## Abstract 
Implicit Neural Representations (INRs) conceptualize videos as neural networks and have recently demonstrated potential as a simple yet promising solution for video compression, converting the intricate issue of video compression into model compression. Their efficiency and fast decoding abilities hold the potential to replace traditional compression methods. However, current approaches are predominantly confined to the pixel, image, or patch-wise levels, failing to fully exploit spatial-temporal information and struggling with the reconstruction of video frames from fixed and time-agnostic embeddings. Addressing these limitations, we introduce Cube-wise Neural Representation for Videos (3DNeRV), where cubes are generated from a learnable 3D encoder, to capture spatiotemporal information in video and use it as an embedding input for the decoder. Additionally, we propose 3DNeRV Block, capable of extracting temporal and spatial features from the cube embedding using Convolution 3D, which enables fast encoding and high-resolution video reconstruction.

| <img src="https://github.com/tuanlda78202/nerv3d/blob/main/utils/assets/model.png" width="800"/> | 
|:--:| 
|3DNeRV architecture |

## Folder Structure

```
3dnerv/
  │── configs/ 
  │   ├── parse_config.py
  │   ├── uvg/
  │   └── hevc-b/ 
  │  
  │── scripts/
  │   ├── /*.sh
  │   ├── train.py
  │   ├── test.py 
  │   ├── compress.py 
  │   └── fps.py 
  │
  │── src/
  │   │── backbone/
  │   │    └── videomae.py
  │   │── compression/
  │   │    └── utils.py
  │   ├── dataset/ 
  │   │    ├── build.py
  │   │    └── yuv.py
  │   ├── evaluation/ 
  │   │    ├── loss.py
  │   │    └── metric.py  
  │   ├── model/ 
  │   │    └── nerv3d.py
  │   └── trainer/ 
  │        ├── base_trainer.py
  │        └── nerv3d_trainer.py
  │ 
  └── utils/
      ├── macs.py
      └── util.py
```

## Usage

### Installation

1. Clone the repository
```bash
git clone https://github.com/tuanlda78202/3dnerv.git
cd 3dnerv
```
2. Install the required packages:
```
pip install -r requirements.txt
```
<!-- pipreqs for get requirements.txt -->

For `NNCodec` with compression task, you need install follow these instructions:
```bash
git clone https://github.com/fraunhoferhhi/nncodec
cd nncodec
pip install wheel
pip install -r requirements.txt
pip install .
```
### Configuration

<details>

<summary>YAML Format</summary>

```yaml
dataloader:
  type: build_data

  args:
    data_path: "../uvg/beauty.yuv"
    batch_size: 1
    frame_interval: 2                   

arch:
  type: NeRV3D

  args:    
    embed_dim: 8
    embed_size: [9, 16]
    decode_dim: 140

    lower_kernel: 1
    upper_kernel: 5
    scales: [5, 3, 2, 2, 2]
    reduce: 1.2
    lower_width: 12

    ckpt_path: "../uvg/pretrain_vit-b_k400_e1600.pth"

loss:
  type: loss_fn

  args:
    loss_type: "L1-SSIM"

psnr:
  type: psnr_batch
  args:

msssim:
  type: msssim_batch
  args:

optimizer:
  type: Adam

  args:  
    lr: 0.002
    betas: [0.9, 0.99]

  lr_encoder: 0.0005

lr_scheduler:
  type: OneCycleLR

  args:
    max_lr: [0.0005, 0.002, 0.002, 0.002] 
    total_steps: 90000

trainer:  
  epochs: 300
  valid_period: 100
  save_dir: saved/
  save_period: 100

  mode: "offline" 
  name: "beauty_12m"
  
  project: 3dnerv-cvpr24
  entity: tuanlda78202
  api_key_file: "./config/api/tuanlda78202"

compression:
  compress_dir: compression/12m/beauty/
  embedding_path: "compression/12m/beauty/compressed_embedding.ncc"
  raw_decoder_path: "compression/12m/beauty/raw_decoder.pt"
  stream_path: "compression/12m/beauty/compressed_decoder.nnc"
  compressed_decoder_path: "compression/12m/beauty/compressed_decoder_converted.pt"

  model_qp: -38
  embed_qp: -38                        
```

</details>

### Training
Modify the configurations in `.yaml` config files, then run:

```bash
python scripts/train.py -c [CONFIG]
```
### Testing
```bash
python scripts/test.py -c [CONFIG] -r [CKPT]
```

### Compression
```bash
python scripts/compess.py -c [CONFIG] -r [CKPT]
```

### Decoding
```bash
python scripts/fps.py -c [CONFIG] -r [CKPT]
```
## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{tuan3dnerv2024,
      title={3DNeRV: A Cube-wise Neural Representations for Videos}, 
      author={Tuan LDA, Minh Nguyen and Thang Nguyen},
      year={2024},
      booktitle={CVPR},
}
```

## Contact
If you have any questions, please feel free to email the [authors.](tuanleducanh78202@gmail.com)

## Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>
