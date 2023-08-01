#!/bin/bash

# Beauty
python scripts/compress.py --config config/uvg-1080p/12M/beauty.yaml --resume saved/models/beauty-1080p_12M/300e.pth

# HoneyBee
python scripts/compress.py --config config/uvg-1080p/12M/bee.yaml --resume saved/models/bee-1080p_12M/300e.pth

# Bosphorus
python scripts/compress.py --config config/uvg-1080p/12M/bosphorus.yaml --resume saved/models/bosphorus-1080p_12M/300e.pth

# Jockey
python scripts/compress.py --config config/uvg-1080p/12M/jockey.yaml --resume saved/models/jockey-1080p_12M/300e.pth

# ReadySetGo
python scripts/compress.py --config config/uvg-1080p/12M/ready.yaml --resume saved/models/ready-1080p_12M/300e.pth

# ShakenDry
python scripts/compress.py --config config/uvg-1080p/12M/shake.yaml --resume saved/models/shake-1080p_12M/300e.pth

# YachtRide
python scripts/compress.py --config config/uvg-1080p/12M/yacht.yaml --resume saved/models/yacht-1080p_12M/300e.pth
