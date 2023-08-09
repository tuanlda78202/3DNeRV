# 3DNeRV

> "3DNeRV: A Cube-wise Neural Representaion for Videos" (accepted at AAAI'24) <br>
> [Tuan LDA](https://tuanlda78202.github.io)<sup>1, 2</sup>, [Minh Nguyen](https://github.com/minhngt62)<sup>1, 2</sup>, [Thang Nguyen](https://scholar.google.com/citations?user=1NjryzEAAAAJ&hl=en&oi=ao)<sup>2</sup><br>
> <sup>1</sup>Hanoi University of Science and Technology, <sup>2</sup>Viettel High Tech<br>

- [3DNeRV](#3dnerv)
  - [Abstract](#abstract)
  - [Folder Structure](#folder-structure)
  - [Usage](#usage)
    - [Installation](#installation)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Testing](#testing)
    - [Compress](#compress)
    - [Decoding](#decoding)
  - [Citation](#citation)
  - [Contact](#contact)
  - [Contributors](#contributors)

## Abstract 
Implicit Neural Representations (INRs) conceptualize videos as neural networks and have recently demonstrated potential as an simple yet promising solution for video compression, converting the intricate issue of video compression into model compression. Their efficiency and fast decoding abilities hold the potential to replace traditional compression methods. However, current approaches are predominantly confined to the pixel, image, or patch-wise levels, and failing to fully exploit spatio-temporal information and struggling with the reconstruction of video frames from fixed and time-agnostic embeddings. Addressing these limitations, we introduce Cube-wise Neural Representation for Videos (3DNeRV) in this paper, where cubes generated from a learnable 3D encoder, to capture spatio-temporal information in video data and uses it as an embedding input for the decoder. Additionally, we propose 3DNeRV-Block, capable of extracting rich temporal and spatial features from the cube embedding using Conv3D, enables fast encoding and high-resolution video reconstruction, particularly excelling in the extraction of temporal information across frames. Our method has been evaluated on the UVG dataset and significantly outperforms all previous INR methods for video reconstruction tasks, achieves state-of-the-art results with significant improvement (+7.11 PSNR increase over HNeRV). We highlight the effectiveness and simplicity of our approach by applying it to DeepCABAC, dominates all previous neural video compression methods, resulting in a 54.82\% bitrate saving over VTM 15.0. This marks a milestone as the first INRs method to outperform deep neural video compression methods and mainstream video coding standards, including H.266. Moreover, our method showcases advantages in decoding, scalability, and adaptability across various deployment scenarios.

| <img src="https://github.com/tuanlda78202/nerv3d/blob/main/utils/assets/model.png" width="800"/> | 
|:--:| 
|3DNeRV architecture |

## Folder Structure

```
nerv3d/
  │── configs/ 
  │   ├── README.md - config name style
  │   ├── parse_config.py
  │   ├── uvg-720p/ 
  │   └── uvg-1080p/ 
  │
  │── data/
  │   ├── *.mp4 - UVG FHD dataset
  │   └── vmae_sdg.pth - VideoMAE ViT-Small distill from Giant
  │
  │── scripts/
  │   ├── */ - Bash scripts for training, testing and compress
  │   ├── train.py 
  │   └── test.py 
  │
  │── src/
  │   │── backbone/
  │   │    └── videomaev2.py
  │   ├── dataset/ 
  │   │    ├── build.py
  │   │    └── datasets.py
  │   ├── evaluation/ 
  │   │    ├── loss.py
  │   │    └── metrics.py  
  │   ├── model/ 
  │   │    └── nerv3d.py
  │   └── trainer/ 
  │        ├── base_trainer.py
  │        └── nerv3d_trainer.py
  │ 
  └── utils/
      ├── log/ 
      └── utils.py
```

## Usage

### Installation

1. Clone the repository
```bash
git clone https://github.com/tuanlda78202/nerv3d.git
cd nerv3d
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
# Intel Xeon Platinum 9282 + NVIDIA A100-PCIE-40GB Config
dataloader:
  type: build_dataloader

  args:
    name: "uvghd30"
    data_path: "data/beauty.mp4"                       
    crop_size: [720, 1280]                             
    num_workers: 6                                     # CPU

    batch_size: 2                                      # MEMORY
    frame_interval: 4                                  # MEMORY

metrics:
  type: psnr_batch

  args:
    batch_size: 2                                   # MEMORY  
    frame_interval: 4                               # MEMORY

arch:
  type: NeRV3D

  args:
    img_size: [720, 1280] 
    frame_interval: 4                   # MEMORY      
    
    embed_dim: 8 
    embed_size: [9, 16]
    decode_dim: 662

    lower_kernel: 1
    upper_kernel: 5
    scales: [5, 4, 2, 2]
    reduce: 3
    lower_width: 6

    ckpt_path: "data/vmae_sdg.pth"

loss:
  type: loss_fn

  args:
    loss_type: "L2"
    batch_average: False

optimizer:
  type: Adam

  args: 
    lr: 0.001 
    betas: [0.9, 0.99]

lr_scheduler:
  type: CosineAnnealingLR

  args:
    T_max: 20000     
    eta_min: 0.000001

trainer:
  resume: False 
  
  epochs: 300
  valid_period: 10

  save_dir: saved/
  save_period: 100
  verbosity: 1

  visual_tool: wandb
  mode: "online"
  project: nerv3d
  api_key_file: "./config/api/tuanlda78202"
  entity: tuanlda78202
  name: "beauty-720p_12M"                         
```

</details>

### Training
Modify the configurations in `.yaml` config files, then run:

```bash
python scripts/train.py --config [CONFIG]
```
### Testing
```bash
python scripts/test.py --config [CONFIG] --resume [CKPT]
```

### Compress
```bash
python scripts/compess.py --config [CONFIG] --resume [CKPT]
```

### Decoding
```bash
python scripts/decoding.py --config [CONFIG] --resume [CKPT]
```
## Citation
If you find our work useful in your research, please cite:
```
@InProceedings{tuan3dnerv2024,
      title={3DNeRV}: Neural Representations for Videos}, 
      author={Tuan LDA, Minh Nguyen and Thang Nguyen},
      year={2024},
      booktitle={AAAI},
}
```

## Contact
If you have any questions, please feel free to email the [authors.](tuan.lda204929@sis.edu.vn)

## Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>
