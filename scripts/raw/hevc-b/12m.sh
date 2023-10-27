#!/bin/bash

# BasketballDrive
python scripts/train.py -c config/uvg/12m/basketball.yaml
python scripts/test.py -c config/uvg/12m/basketball.yaml -r saved/models/basketball_12m/300e.pth

# Cactus
python scripts/train.py -c config/uvg/12m/cactus.yaml
python scripts/test.py -c config/uvg/12m/cactus.yaml -r saved/models/cactus_12m/300e.pth

# Kimono
python scripts/train.py -c config/uvg/12m/kimono.yaml
python scripts/test.py -c config/uvg/12m/kimono.yaml -r saved/models/kimono_12m/300e.pth

# ParkScene
python scripts/train.py -c config/uvg/12m/park.yaml
python scripts/test.py -c config/uvg/12m/park.yaml -r saved/models/park_12m/300e.pth

# BQTerrace
python scripts/train.py -c config/uvg/12m/terrace.yaml
python scripts/test.py -c config/uvg/12m/terrace.yaml -r saved/models/terrace_12m/300e.pth
