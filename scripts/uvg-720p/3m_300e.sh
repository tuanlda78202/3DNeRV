#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-720p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-720p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Jockey
python scripts/train.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# YachtRide
python scripts/train.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth
