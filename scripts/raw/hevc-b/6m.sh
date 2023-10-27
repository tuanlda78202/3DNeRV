#!/bin/bash

# BasketballDrive
python scripts/train.py -c config/uvg/6m/basketball.yaml
python scripts/test.py -c config/uvg/6m/basketball.yaml -r saved/models/basketball_6m/300e.pth

# Cactus
python scripts/train.py -c config/uvg/6m/cactus.yaml
python scripts/test.py -c config/uvg/6m/cactus.yaml -r saved/models/cactus_6m/300e.pth

# Kimono
python scripts/train.py -c config/uvg/6m/kimono.yaml
python scripts/test.py -c config/uvg/6m/kimono.yaml -r saved/models/kimono_6m/300e.pth

# ParkScene
python scripts/train.py -c config/uvg/6m/park.yaml
python scripts/test.py -c config/uvg/6m/park.yaml -r saved/models/park_6m/300e.pth

# BQTerrace
python scripts/train.py -c config/uvg/6m/terrace.yaml
python scripts/test.py -c config/uvg/6m/terrace.yaml -r saved/models/terrace_6m/300e.pth
