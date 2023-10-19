#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg/3M/beauty.yaml
python scripts/test.py --config config/uvg/3M/beauty.yaml --resume saved/models/beauty-1080p_3M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg/3M/bee.yaml
python scripts/test.py --config config/uvg/3M/bee.yaml --resume saved/models/bee-1080p_3M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg/3M/bosphorus.yaml
python scripts/test.py --config config/uvg/3M/bosphorus.yaml --resume saved/models/bosphorus-1080p_3M/300e.pth

# Jockey
python scripts/train.py --config config/uvg/3M/jockey.yaml
python scripts/test.py --config config/uvg/3M/jockey.yaml --resume saved/models/jockey-1080p_3M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg/3M/ready.yaml
python scripts/test.py --config config/uvg/3M/ready.yaml --resume saved/models/ready-1080p_3M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg/3M/shake.yaml
python scripts/test.py --config config/uvg/3M/shake.yaml --resume saved/models/shake-1080p_3M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg/3M/yacht.yaml
python scripts/test.py --config config/uvg/3M/yacht.yaml --resume saved/models/yacht-1080p_3M/300e.pth
