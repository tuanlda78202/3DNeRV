#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg/12M/beauty.yaml
python scripts/test.py --config config/uvg/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg/12M/bee.yaml
python scripts/test.py --config config/uvg/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg/12M/bosphorus.yaml
python scripts/test.py --config config/uvg/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth

# Jockey
python scripts/train.py --config config/uvg/12M/jockey.yaml
python scripts/test.py --config config/uvg/12M/jockey.yaml --resume saved/models/jockey-1080p_12M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg/12M/ready.yaml
python scripts/test.py --config config/uvg/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg/12M/shake.yaml
python scripts/test.py --config config/uvg/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg/12M/yacht.yaml
python scripts/test.py --config config/uvg/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth
