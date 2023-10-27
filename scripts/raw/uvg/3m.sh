#!/bin/bash

# Beauty
python scripts/train.py -c config/uvg/3m/beauty.yaml
python scripts/test.py -c config/uvg/3m/beauty.yaml -r saved/models/beauty_3m/300e.pth

# HoneyBee
python scripts/train.py -c config/uvg/3m/bee.yaml
python scripts/test.py -c config/uvg/3m/bee.yaml -r saved/models/bee_3m/300e.pth

# Bosphorus
python scripts/train.py -c config/uvg/3m/bosphorus.yaml
python scripts/test.py -c config/uvg/3m/bosphorus.yaml -r saved/models/bosphorus_3m/300e.pth

# Jockey
python scripts/train.py -c config/uvg/3m/jockey.yaml
python scripts/test.py -c config/uvg/3m/jockey.yaml -r saved/models/jockey_3m/300e.pth

# ReadySetGo
python scripts/train.py -c config/uvg/3m/ready.yaml
python scripts/test.py -c config/uvg/3m/ready.yaml -r saved/models/ready_3m/300e.pth

# ShakenDry
python scripts/train.py -c config/uvg/3m/shake.yaml
python scripts/test.py -c config/uvg/3m/shake.yaml -r saved/models/shake_3m/300e.pth

# YachtRide
python scripts/train.py -c config/uvg/3m/yacht.yaml
python scripts/test.py -c config/uvg/3m/yacht.yaml -r saved/models/yacht_3m/300e.pth
