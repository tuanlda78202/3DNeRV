#!/bin/bash

# Beauty
python scripts/train.py --config config/uvg-1080p/12M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/beauty-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/beauty-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# HoneyBee
python scripts/train.py --config config/uvg-1080p/12M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bee-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bee-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Bosphorus
python scripts/train.py --config config/uvg-1080p/12M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/bosphorus-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/bosphorus-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# Jockey
python scripts/train.py --config config/uvg-1080p/12M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/jockey-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/jockey-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ReadySetGo
python scripts/train.py --config config/uvg-1080p/12M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/ready-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/ready-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# ShakenDry
python scripts/train.py --config config/uvg-1080p/12M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/shake-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/shake-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth

# YachtRide
python scripts/train.py --config config/uvg-1080p/12M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
python scripts/test.py --config config/uvg-1080p/12M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/yacht-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/yacht-12M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e-ckpt300e.pth
