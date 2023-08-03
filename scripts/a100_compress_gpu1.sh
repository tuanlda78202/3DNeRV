#!/bin/bash

# qp = -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth --mqp -32 --eqp -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth --mqp -32 --eqp -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth --mqp -32 --eqp -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth --mqp -32 --eqp -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth --mqp -32 --eqp -32
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth --mqp -32 --eqp -32

# qp = -30 (lower bound)
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth --mqp -30 --eqp -30
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth --mqp -30 --eqp -30
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth --mqp -30 --eqp -30
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth --mqp -30 --eqp -30
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth --mqp -30 --eqp -30
CUDA_VISIBLE_DEVICES=1 python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth --mqp -30 --eqp -30
