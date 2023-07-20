#!/bin/bash

# Beauty
echo "Training Beauty 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training Beauty 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for Beauty 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/beauty_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Beauty-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for Beauty 720x1080, model size = 3M from ckpte299.pth"

# HoneyBee
echo "Training HoneyBee 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training HoneyBee 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for HoneyBee 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/bee_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Bee-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for HoneyBee 720x1080, model size = 3M from ckpte299.pth"

# Bosphorus
echo "Training Bosphorus 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training Bosphorus 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for Bosphorus 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Bosphorus-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for Bosphorus 720x1080, model size = 3M from ckpte299.pth"

# Jockey
echo "Training Jockey 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training Jockey 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for Jockey 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Jockey-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for Jockey 720x1080, model size = 3M from ckpte299.pth"

# ReadySetGo
echo "Training ReadySetGo 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training ReadySetGo 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for ReadySetGo 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Ready-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for ReadySetGo 720x1080, model size = 3M from ckpte299.pth"

# ShakenDry
echo "Training ShakenDry 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training ShakenDry 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for ShakenDry 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Shake-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for ShakenDry 720x1080, model size = 3M from ckpte299.pth"

# YachtRide
echo "Training YachtRide 720x1080, model size = 3M, for 300 epochs"
python ../train.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml
echo "Completed training YachtRide 720x1080, model size = 3M, training 300 epochs"
echo "Evaluating PSNR & Inference video for YachtRide 720x1080, model size = 3M from ckpte300.pth"
python ../test.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e.yaml --resume "saved/models/Yacht-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-10k_300e/e299.pth"
echo "Completed caculate PSNR & inference video for YachtRide 720x1080, model size = 3M from ckpte299.pth"