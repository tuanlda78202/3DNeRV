#!/bin/bash

# GMACs
python src/model/macs.py

## FAILED VIDEOS
# Jockey 12M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/12M/jockey.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/12M/jockey.yaml --resume saved/models/jockey-1080p_12M/300e.pth

# Bosphorus 6M
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/uvg-1080p/6M/bosphorus.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth

# Jockey 6M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/6M/jockey.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth

# Bosphorus 3M
python scripts/train.py --config config/uvg-1080p/3M/bosphorus.yaml
python scripts/test.py --config config/uvg-1080p/3M/bosphorus.yaml --resume saved/models/bosphorus-1080p_3M/300e.pth

## 6 video 3M (scripts/train-uvg-1080p/3m_300e.sh)

* 12m
* jockey
* 6m
* bos
* jockey
* 3m
* bee
* bos
* jockey
* yacht
* shake
* ready

## NNC Compression (scripts/compress-uvg-1080p)

## MS-SSIM evaluation (run all exp video rescontruction for re-testing)
