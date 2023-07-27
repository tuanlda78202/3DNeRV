#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-720p/12M/beauty.yaml
python scripts/test.py --config config/uvg-720p/12M/beauty.yaml --resume saved/models/beauty-720p_12M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-720p/12M/bee.yaml
python scripts/test.py --config config/uvg-720p/12M/bee.yaml --resume saved/models/bee-720p_12M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-720p/12M/bosphorus.yaml
python scripts/test.py --config config/uvg-720p/12M/bosphorus.yaml --resume saved/models/bosphorus-720p_12M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-720p/12M/jockey.yaml
python scripts/test.py --config config/uvg-720p/12M/jockey.yaml --resume saved/models/jockey-720p_12M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-720p/12M/ready.yaml
python scripts/test.py --config config/uvg-720p/12M/ready.yaml --resume saved/models/ready-720p_12M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-720p/12M/shake.yaml
python scripts/test.py --config config/uvg-720p/12M/shake.yaml --resume saved/models/shake-720p_12M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-720p/12M/yacht.yaml
python scripts/test.py --config config/uvg-720p/12M/yacht.yaml --resume saved/models/yacht-720p_12M/300e.pth
