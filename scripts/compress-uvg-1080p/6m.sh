#!/bin/bash

# qp = -46 (upper bound)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -46 --eqp -46

# qp = -42
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -42 --eqp -42

# qp = -38 (default)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -38 --eqp -38

# qp = -34
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -34 --eqp -34

# qp = -30 (lower bound)
python scripts/compress.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth --mqp -30 --eqp -30
