#!/bin/bash

### A100 Config ###

######################################################################################################################
## 1080p - 12M

# HoneyBee
#python scripts/train.py --config config/uvg-1080p/12M/bee.yaml
python scripts/test.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M-vmaev2/300e.pth

# Beauty
python scripts/train.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/100e.pth
python scripts/test.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth

# HoneyBee
#python scripts/train.py --config config/uvg-1080p/12M/bee.yaml
# python scripts/test.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M-vmaev2/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/100e.pth
python scripts/test.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/12M/jockey.yaml
python scripts/test.py --config config/uvg-1080p/12M/jockey.yaml --resume saved/models/jockey-1080p_12M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/12M/ready.yaml
python scripts/test.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/12M/shake.yaml
python scripts/test.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/12M/yacht.yaml
python scripts/test.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth

######################################################################################################################
## 1080p - 6M

# Beauty
python scripts/train.py --config config/uvg-1080p/6M/beauty.yaml
python scripts/test.py --config config/uvg-1080p/6M/beauty.yaml --resume saved/models/beauty-1080p_6M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-1080p/6M/bee.yaml
python scripts/test.py --config config/uvg-1080p/6M/bee.yaml --resume saved/models/bee-1080p_6M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/6M/bosphorus.yaml
python scripts/test.py --config config/uvg-1080p/6M/bosphorus.yaml --resume saved/models/bosphorus-1080p_6M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/6M/jockey.yaml
python scripts/test.py --config config/uvg-1080p/6M/jockey.yaml --resume saved/models/jockey-1080p_6M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/6M/ready.yaml
python scripts/test.py --config config/uvg-1080p/6M/ready.yaml --resume saved/models/ready-1080p_6M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/6M/shake.yaml
python scripts/test.py --config config/uvg-1080p/6M/shake.yaml --resume saved/models/shake-1080p_6M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/6M/yacht.yaml
python scripts/test.py --config config/uvg-1080p/6M/yacht.yaml --resume saved/models/yacht-1080p_6M/300e.pth

######################################################################################################################
## 1080p - 3M

# Beauty
python scripts/train.py --config config/uvg-1080p/3M/beauty.yaml
python scripts/test.py --config config/uvg-1080p/3M/beauty.yaml --resume saved/models/beauty-1080p_3M/300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-1080p/3M/bee.yaml
python scripts/test.py --config config/uvg-1080p/3M/bee.yaml --resume saved/models/bee-1080p_3M/300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/3M/bosphorus.yaml
python scripts/test.py --config config/uvg-1080p/3M/bosphorus.yaml --resume saved/models/bosphorus-1080p_3M/300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/3M/jockey.yaml
python scripts/test.py --config config/uvg-1080p/3M/jockey.yaml --resume saved/models/jockey-1080p_3M/300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/3M/ready.yaml
python scripts/test.py --config config/uvg-1080p/3M/ready.yaml --resume saved/models/ready-1080p_3M/300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/3M/shake.yaml
python scripts/test.py --config config/uvg-1080p/3M/shake.yaml --resume saved/models/shake-1080p_3M/300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/3M/yacht.yaml
python scripts/test.py --config config/uvg-1080p/3M/yacht.yaml --resume saved/models/yacht-1080p_3M/300e.pth
