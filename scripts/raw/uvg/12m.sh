#!/bin/bash

# Beauty
python scripts/train.py -c config/uvg/12m/beauty.yaml
python scripts/test.py -c config/uvg/12m/beauty.yaml -r saved/models/beauty_12m/300e.pth

# HoneyBee
python scripts/train.py -c config/uvg/12m/bee.yaml
python scripts/test.py -c config/uvg/12m/bee.yaml -r saved/models/bee_12m/300e.pth

# Bosphorus
python scripts/train.py -c config/uvg/12m/bosphorus.yaml
python scripts/test.py -c config/uvg/12m/bosphorus.yaml -r saved/models/bosphorus_12m/300e.pth

# Jockey
python scripts/train.py -c config/uvg/12m/jockey.yaml
python scripts/test.py -c config/uvg/12m/jockey.yaml -r saved/models/jockey_12m/300e.pth

# ReadySetGo
python scripts/train.py -c config/uvg/12m/ready.yaml
python scripts/test.py -c config/uvg/12m/ready.yaml -r saved/models/ready_12m/300e.pth

# ShakenDry
python scripts/train.py -c config/uvg/12m/shake.yaml
python scripts/test.py -c config/uvg/12m/shake.yaml -r saved/models/shake_12m/300e.pth

# YachtRide
python scripts/train.py -c config/uvg/12m/yacht.yaml
python scripts/test.py -c config/uvg/12m/yacht.yaml -r saved/models/yacht_12m/300e.pth
