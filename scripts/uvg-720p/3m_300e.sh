#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-720p/3M/beauty.yaml
python scripts/test.py --config config/uvg-720p/3M/beauty.yaml --resume saved/models/beauty-720p_3M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-720p/3M/bee.yaml
python scripts/test.py --config config/uvg-720p/3M/bee.yaml --resume saved/models/bee-720p_3M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-720p/3M/bosphorus.yaml
python scripts/test.py --config config/uvg-720p/3M/bosphorus.yaml --resume saved/models/bosphorus-720p_3M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-720p/3M/jockey.yaml
python scripts/test.py --config config/uvg-720p/3M/jockey.yaml --resume saved/models/jockey-720p_3M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-720p/3M/ready.yaml
python scripts/test.py --config config/uvg-720p/3M/ready.yaml --resume saved/models/ready-720p_3M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-720p/3M/shake.yaml
python scripts/test.py --config config/uvg-720p/3M/shake.yaml --resume saved/models/shake-720p_3M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-720p/3M/yacht.yaml
python scripts/test.py --config config/uvg-720p/3M/yacht.yaml --resume saved/models/yacht-720p_3M/300e.pth
