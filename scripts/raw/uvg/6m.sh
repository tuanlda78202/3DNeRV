#!/bin/bash

# Beauty
python scripts/train.py -c config/uvg/6m/beauty.yaml
python scripts/test.py -c config/uvg/6m/beauty.yaml -r saved/models/beauty_6m/300e.pth

# HoneyBee
python scripts/train.py -c config/uvg/6m/bee.yaml
python scripts/test.py -c config/uvg/6m/bee.yaml -r saved/models/bee_6m/300e.pth

# Bosphorus
python scripts/train.py -c config/uvg/6m/bosphorus.yaml
python scripts/test.py -c config/uvg/6m/bosphorus.yaml -r saved/models/bosphorus_6m/300e.pth

# Jockey
python scripts/train.py -c config/uvg/6m/jockey.yaml
python scripts/test.py -c config/uvg/6m/jockey.yaml -r saved/models/jockey_6m/300e.pth

# ReadySetGo
python scripts/train.py -c config/uvg/6m/ready.yaml
python scripts/test.py -c config/uvg/6m/ready.yaml -r saved/models/ready_6m/300e.pth

# ShakenDry
python scripts/train.py -c config/uvg/6m/shake.yaml
python scripts/test.py -c config/uvg/6m/shake.yaml -r saved/models/shake_6m/300e.pth

# YachtRide
python scripts/train.py -c config/uvg/6m/yacht.yaml
python scripts/test.py -c config/uvg/6m/yacht.yaml -r saved/models/yacht_6m/300e.pth
