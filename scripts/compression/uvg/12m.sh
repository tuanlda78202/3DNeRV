#!/bin/bash

# qp = -46 (upper bound)
python scripts/compress.py --config config/uvg/12m/beauty.yaml --resume saved/models/beauty_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/bee.yaml --resume saved/models/bee_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/bosphorus.yaml --resume saved/models/bosphorus_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/jockey.yaml --resume saved/models/jockey_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/ready.yaml --resume saved/models/ready_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/shake.yaml --resume saved/models/shake_12m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/12m/yacht.yaml --resume saved/models/yacht_12m/300e.pth --mqp -46 --eqp -46

# qp = -42
python scripts/compress.py --config config/uvg/12m/beauty.yaml --resume saved/models/beauty_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/bee.yaml --resume saved/models/bee_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/bosphorus.yaml --resume saved/models/bosphorus_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/jockey.yaml --resume saved/models/jockey_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/ready.yaml --resume saved/models/ready_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/shake.yaml --resume saved/models/shake_12m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/12m/yacht.yaml --resume saved/models/yacht_12m/300e.pth --mqp -42 --eqp -42

# qp = -38 (default)
python scripts/compress.py --config config/uvg/12m/beauty.yaml --resume saved/models/beauty_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/bee.yaml --resume saved/models/bee_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/bosphorus.yaml --resume saved/models/bosphorus_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/jockey.yaml --resume saved/models/jockey_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/ready.yaml --resume saved/models/ready_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/shake.yaml --resume saved/models/shake_12m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/12m/yacht.yaml --resume saved/models/yacht_12m/300e.pth --mqp -38 --eqp -38

# qp = -34
python scripts/compress.py --config config/uvg/12m/beauty.yaml --resume saved/models/beauty_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/bee.yaml --resume saved/models/bee_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/bosphorus.yaml --resume saved/models/bosphorus_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/jockey.yaml --resume saved/models/jockey_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/ready.yaml --resume saved/models/ready_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/shake.yaml --resume saved/models/shake_12m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/12m/yacht.yaml --resume saved/models/yacht_12m/300e.pth --mqp -34 --eqp -34

# qp = -30 (lower bound)
python scripts/compress.py --config config/uvg/12m/beauty.yaml --resume saved/models/beauty_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/bee.yaml --resume saved/models/bee_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/bosphorus.yaml --resume saved/models/bosphorus_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/jockey.yaml --resume saved/models/jockey_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/ready.yaml --resume saved/models/ready_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/shake.yaml --resume saved/models/shake_12m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/12m/yacht.yaml --resume saved/models/yacht_12m/300e.pth --mqp -30 --eqp -30
