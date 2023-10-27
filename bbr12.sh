#!/bin/bash

# Beauty
python scripts/train.py -c config/uvg/12m/beauty.yaml
python scripts/test.py -c config/uvg/12m/beauty.yaml -r saved/models/beauty_12m/300e.pth

# HoneyBee
python scripts/train.py -c config/uvg/12m/bee.yaml
python scripts/test.py -c config/uvg/12m/bee.yaml -r saved/models/bee_12m/300e.pth

# ReadySetGo
python scripts/train.py -c config/uvg/12m/ready.yaml
python scripts/test.py -c config/uvg/12m/ready.yaml -r saved/models/ready_12m/300e.pth
