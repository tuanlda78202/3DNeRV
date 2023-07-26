#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-1080p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/beauty-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-1080p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bee-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bee-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bosphorus-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bosphorus-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/jockey-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/jockey-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/ready-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/ready-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/shake-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/shake-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/yacht-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/yacht-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth
