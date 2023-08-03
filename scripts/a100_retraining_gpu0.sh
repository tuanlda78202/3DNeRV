#!/bin/bash

# Jockey 12M
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/uvg-1080p/12M/jockey.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --config config/uvg-1080p/12M/jockey.yaml --resume saved/models/jockey-1080p_12M/300e.pth

# Bosphorus 6M
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/uvg-1080p/6M/bosphorus.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth

# Bosphorus 3M
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/uvg-1080p/3M/bosphorus.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --config config/uvg-1080p/3M/bosphorus.yaml --resume saved/models/bosphorus-1080p_3M/300e.pth

# Jockey 3M
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config config/uvg-1080p/3M/jockey.yaml
CUDA_VISIBLE_DEVICES=0 python scripts/test.py --config config/uvg-1080p/3M/jockey.yaml --resume saved/models/jockey-1080p_3M/300e.pth
