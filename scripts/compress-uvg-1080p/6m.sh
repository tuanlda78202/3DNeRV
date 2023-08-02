#!/bin/bash

# qp = -50
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -50 --eqp -50
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -50 --eqp -50

# qp = -48 (upper bound)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -48 --eqp -48
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -48 --eqp -48

# qp = -38 (default)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -38 --eqp -38

# qp = -32
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -32 --eqp -32
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -32 --eqp -32

# qp = -30 (lower bound)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -30 --eqp -30
