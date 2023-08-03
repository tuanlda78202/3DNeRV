#!/bin/bash

# GMACs
CUDA_VISIBLE_DEVICES=0 python src/model/macs.py

# 12M
# qp = -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth --mqp -50 --eqp -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth --mqp -50 --eqp -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth --mqp -50 --eqp -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth --mqp -50 --eqp -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth --mqp -50 --eqp -50
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth --mqp -50 --eqp -50

# qp = -48 (upper bound)
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth --mqp -48 --eqp -48
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth --mqp -48 --eqp -48
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth --mqp -48 --eqp -48
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth --mqp -48 --eqp -48
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth --mqp -48 --eqp -48
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth --mqp -48 --eqp -48

# qp = -38 (default)
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth --mqp -38 --eqp -38
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth --mqp -38 --eqp -38
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth --mqp -38 --eqp -38
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth --mqp -38 --eqp -38
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth --mqp -38 --eqp -38
CUDA_VISIBLE_DEVICES=0 python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth --mqp -38 --eqp -38
