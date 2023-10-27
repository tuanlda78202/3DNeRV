#!/bin/bash

# qp = -46 (upper bound)
python scripts/compress.py --config config/uvg/3m/beauty.yaml --resume saved/models/beauty_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/bee.yaml --resume saved/models/bee_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/bosphorus.yaml --resume saved/models/bosphorus_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/jockey.yaml --resume saved/models/jockey_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/ready.yaml --resume saved/models/ready_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/shake.yaml --resume saved/models/shake_3m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/3m/yacht.yaml --resume saved/models/yacht_3m/300e.pth --mqp -46 --eqp -46

# qp = -42
python scripts/compress.py --config config/uvg/3m/beauty.yaml --resume saved/models/beauty_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/bee.yaml --resume saved/models/bee_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/bosphorus.yaml --resume saved/models/bosphorus_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/jockey.yaml --resume saved/models/jockey_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/ready.yaml --resume saved/models/ready_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/shake.yaml --resume saved/models/shake_3m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/3m/yacht.yaml --resume saved/models/yacht_3m/300e.pth --mqp -42 --eqp -42

# qp = -38 (default)
python scripts/compress.py --config config/uvg/3m/beauty.yaml --resume saved/models/beauty_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/bee.yaml --resume saved/models/bee_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/bosphorus.yaml --resume saved/models/bosphorus_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/jockey.yaml --resume saved/models/jockey_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/ready.yaml --resume saved/models/ready_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/shake.yaml --resume saved/models/shake_3m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/3m/yacht.yaml --resume saved/models/yacht_3m/300e.pth --mqp -38 --eqp -38

# qp = -34
python scripts/compress.py --config config/uvg/3m/beauty.yaml --resume saved/models/beauty_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/bee.yaml --resume saved/models/bee_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/bosphorus.yaml --resume saved/models/bosphorus_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/jockey.yaml --resume saved/models/jockey_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/ready.yaml --resume saved/models/ready_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/shake.yaml --resume saved/models/shake_3m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/3m/yacht.yaml --resume saved/models/yacht_3m/300e.pth --mqp -34 --eqp -34

# qp = -30 (lower bound)
python scripts/compress.py --config config/uvg/3m/beauty.yaml --resume saved/models/beauty_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/bee.yaml --resume saved/models/bee_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/bosphorus.yaml --resume saved/models/bosphorus_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/jockey.yaml --resume saved/models/jockey_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/ready.yaml --resume saved/models/ready_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/shake.yaml --resume saved/models/shake_3m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/3m/yacht.yaml --resume saved/models/yacht_3m/300e.pth --mqp -30 --eqp -30
