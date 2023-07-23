# 3DNeRV
- [3DNeRV](#3dnerv)
  - [Folder Structure](#folder-structure)
  - [Model Zoo](#model-zoo)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Testing](#testing)
  - [Contributors](#contributors)

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
  │   ├── uvg/ - Bash scripts auto training & testing 
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
  │   │    ├── hnerv3d.py
  │   │    └── macs.py  
  │   └── trainer/ 
  │        ├── base_trainer.py
  │        └── nerv3d_trainer.py
  │ 
  └── utils/
      ├── log/ 
      └── utils.py
```
## Model Zoo 
<summary></summary>

<table style="margin-left:auto;margin-right:auto;font-size:1.4vw;padding:10px 10px;text-align:center;vertical-align:center;">
  <tr>
    <td colspan="7" style="font-weight:bold;">Neural Representation for Videos</td>
  </tr>
  <tr>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/u2net/README.md">NeRV</a> (NIPS'21)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">E-NeRV</a> (ECCV'22)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">FFNeRV</a> (arXiv'22)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">D-NeRV</a> (CVPR'23)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">HNeRV</a> (CVPR'23)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">HiNeRV</a> (NIPS'23)</td>
    <td><a href="https://github.com/tuanlda78202/CVP/blob/main/configs/dis/README.md">NeRV3D</a> (Ours)</td>
  </tr>
</table>

## Usage

If you run private repository on online notebook:
1. [Generate your token](https://github.com/settings/tokens)
2. Get repo address from `github.com/.../...git`: 
```bash
git clone https://your_personal_token@github.com/tuanlda78202/nerv3d.git
cd nerv3d
```
3. Install the required packages:
```
pip install -r requirements.txt
```
<!-- pipreqs for get requirements.txt -->
### Configuration

<details>

<summary>YAML Format</summary>

```yaml
name: Beauty-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e

dataloader:
  type: build_dataloader

  args:
    name: "uvghd30"
    data_path: "data/beauty.mp4"                       
    crop_size: [720, 1280]                             
    num_workers: 6  

    batch_size: 2                                      
    frame_interval: 4                                  

metrics:
  type: psnr_batch

  args:
    batch_size: 2                                     
    frame_interval: 4                               

arch:
  type: HNeRVMae

  args:
    img_size: [720, 1280] 
    frame_interval: 4                        
    
    embed_dim: 8 
    embed_size: [9, 16]
    decode_dim: 314

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
    T_max: 0.000001     
    eta_min: 0.0

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
  name: "beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e"              
```

</details>

### Training
Modify the configurations in `.yaml` config files, then run:

```bash
python scripts/train.py --config [CONFIG]
```
If you want training and testing multiple videos, run:
```bash
sh scripts/pipleline.sh
```

### Testing
```bash
python scripts/test.py --config [CONFIG]
```

## Contributors 
<a href="https://github.com/tuanlda78202/MLR/graphs/contributors">
<img src="https://contrib.rocks/image?repo=tuanlda78202/MLR" /></a>
</a>
