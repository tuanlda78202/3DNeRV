#!/bin/bash

# qp = -46 (upper bound)
python scripts/compress.py --config config/uvg/6m/beauty.yaml --resume saved/models/beauty_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/bee.yaml --resume saved/models/bee_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/bosphorus.yaml --resume saved/models/bosphorus_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/jockey.yaml --resume saved/models/jockey_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/ready.yaml --resume saved/models/ready_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/shake.yaml --resume saved/models/shake_6m/300e.pth --mqp -46 --eqp -46
python scripts/compress.py --config config/uvg/6m/yacht.yaml --resume saved/models/yacht_6m/300e.pth --mqp -46 --eqp -46

# qp = -42
python scripts/compress.py --config config/uvg/6m/beauty.yaml --resume saved/models/beauty_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/bee.yaml --resume saved/models/bee_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/bosphorus.yaml --resume saved/models/bosphorus_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/jockey.yaml --resume saved/models/jockey_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/ready.yaml --resume saved/models/ready_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/shake.yaml --resume saved/models/shake_6m/300e.pth --mqp -42 --eqp -42
python scripts/compress.py --config config/uvg/6m/yacht.yaml --resume saved/models/yacht_6m/300e.pth --mqp -42 --eqp -42

# qp = -38 (default)
python scripts/compress.py --config config/uvg/6m/beauty.yaml --resume saved/models/beauty_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/bee.yaml --resume saved/models/bee_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/bosphorus.yaml --resume saved/models/bosphorus_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/jockey.yaml --resume saved/models/jockey_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/ready.yaml --resume saved/models/ready_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/shake.yaml --resume saved/models/shake_6m/300e.pth --mqp -38 --eqp -38
python scripts/compress.py --config config/uvg/6m/yacht.yaml --resume saved/models/yacht_6m/300e.pth --mqp -38 --eqp -38

# qp = -34
python scripts/compress.py --config config/uvg/6m/beauty.yaml --resume saved/models/beauty_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/bee.yaml --resume saved/models/bee_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/bosphorus.yaml --resume saved/models/bosphorus_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/jockey.yaml --resume saved/models/jockey_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/ready.yaml --resume saved/models/ready_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/shake.yaml --resume saved/models/shake_6m/300e.pth --mqp -34 --eqp -34
python scripts/compress.py --config config/uvg/6m/yacht.yaml --resume saved/models/yacht_6m/300e.pth --mqp -34 --eqp -34

# qp = -30 (lower bound)
python scripts/compress.py --config config/uvg/6m/beauty.yaml --resume saved/models/beauty_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/bee.yaml --resume saved/models/bee_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/bosphorus.yaml --resume saved/models/bosphorus_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/jockey.yaml --resume saved/models/jockey_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/ready.yaml --resume saved/models/ready_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/shake.yaml --resume saved/models/shake_6m/300e.pth --mqp -30 --eqp -30
python scripts/compress.py --config config/uvg/6m/yacht.yaml --resume saved/models/yacht_6m/300e.pth --mqp -30 --eqp -30
