#!/bin/bash

# BasketballDrive
python scripts/train.py -c config/uvg/3m/basketball.yaml
python scripts/test.py -c config/uvg/3m/basketball.yaml -r saved/models/basketball_3m/300e.pth

# Cactus
python scripts/train.py -c config/uvg/3m/cactus.yaml
python scripts/test.py -c config/uvg/3m/cactus.yaml -r saved/models/cactus_3m/300e.pth

# Kimono
python scripts/train.py -c config/uvg/3m/kimono.yaml
python scripts/test.py -c config/uvg/3m/kimono.yaml -r saved/models/kimono_3m/300e.pth

# ParkScene
python scripts/train.py -c config/uvg/3m/park.yaml
python scripts/test.py -c config/uvg/3m/park.yaml -r saved/models/park_3m/300e.pth

# BQTerrace
python scripts/train.py -c config/uvg/3m/terrace.yaml
python scripts/test.py -c config/uvg/3m/terrace.yaml -r saved/models/terrace_3m/300e.pth
