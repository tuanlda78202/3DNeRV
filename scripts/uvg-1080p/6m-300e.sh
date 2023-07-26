#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-1080p/6M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/beauty-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/beauty-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-1080p/6M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bee-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bee-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/6M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bosphorus-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bosphorus-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/6M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/jockey-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/jockey-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/6M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/ready-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/ready-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/6M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/shake-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/shake-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/6M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/6M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/yacht-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/yacht-6M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth
