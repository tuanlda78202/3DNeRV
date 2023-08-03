#!/bin/bash

# Jockey 6M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/6M/jockey.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth

# HoneyBee 3M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/3M/bee.yaml --resume saved/models/bee-1080p_3M/100e.pth
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/3M/bee.yaml --resume saved/models/bee-1080p_3M/300e.pth

# ReadySetGo 3M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/3M/ready.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/3M/ready.yaml --resume saved/models/ready-1080p_3M/300e.pth

# ShakenDry 3M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/3M/shake.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/3M/shake.yaml --resume saved/models/shake-1080p_3M/300e.pth

# YachtRide 3M
CUDA_VISIBLE_DEVICES=1 python scripts/train.py --config config/uvg-1080p/3M/yacht.yaml
CUDA_VISIBLE_DEVICES=1 python scripts/test.py --config config/uvg-1080p/3M/yacht.yaml --resume saved/models/yacht-1080p_3M/300e.pth
