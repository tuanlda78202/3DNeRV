#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-720p/6M/beauty.yaml
python scripts/test.py --config config/uvg-720p/6M/beauty.yaml --resume saved/models/beauty-720p_6M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-720p/6M/bee.yaml
python scripts/test.py --config config/uvg-720p/6M/bee.yaml --resume saved/models/bee-720p_6M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-720p/6M/bosphorus.yaml
python scripts/test.py --config config/uvg-720p/6M/bosphorus.yaml --resume saved/models/bosphorus-720p_6M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-720p/6M/jockey.yaml
python scripts/test.py --config config/uvg-720p/6M/jockey.yaml --resume saved/models/jockey-720p_6M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-720p/6M/ready.yaml
python scripts/test.py --config config/uvg-720p/6M/ready.yaml --resume saved/models/ready-720p_6M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-720p/6M/shake.yaml
python scripts/test.py --config config/uvg-720p/6M/shake.yaml --resume saved/models/shake-720p_6M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-720p/6M/yacht.yaml
python scripts/test.py --config config/uvg-720p/6M/yacht.yaml --resume saved/models/yacht-720p_6M/300e.pth
