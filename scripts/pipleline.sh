# Bash scripts pipeline training and testing for 7 videos in UVG dataset

### 3M-720p

# Bosphorus
echo "Training Bosphorus 720x1080, model size = 3M, for 300 epochs"
python scripts/train.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
echo "Completed training Bosphorus 720x1080, model size = 3M, training 300 epochs"

echo "Evaluating PSNR & Inference video for Bosphorus 720x1080, model size = 3M from ckpte300.pth"
python scripts/test.py --config config/uvg-720p/3M/bosphorus_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/Bosphorus-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth
echo "Completed caculate PSNR & inference video for Bosphorus 720x1080, model size = 3M from ckpte299.pth"

mv saved/models/Bosphorus-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth ../ckpt/720p/bosphorus-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_ckpt300e.pth

# Jockey
echo "Training Jockey 720x1080, model size = 3M, for 300 epochs"
python scripts/train.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
echo "Completed training Jockey 720x1080, model size = 3M, training 300 epochs"

echo "Evaluating PSNR & Inference video for Jockey 720x1080, model size = 3M from ckpte300.pth"
python scripts/test.py --config config/uvg-720p/3M/jockey_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/Jockey-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth
echo "Completed caculate PSNR & inference video for Jockey 720x1080, model size = 3M from ckpte299.pth"

mv saved/models/Jockey-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth ../ckpt/720p/jockey-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_ckpt300e.pth

# ShakenDry
echo "Training ShakenDry 720x1080, model size = 3M, for 300 epochs"
python scripts/train.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
echo "Completed training ShakenDry 720x1080, model size = 3M, training 300 epochs"

echo "Evaluating PSNR & Inference video for ShakenDry 720x1080, model size = 3M from ckpte300.pth"
python scripts/test.py --config config/uvg-720p/3M/shake_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/Shake-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth
echo "Completed caculate PSNR & inference video for ShakenDry 720x1080, model size = 3M from ckpte299.pth"

mv saved/models/Shake-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth ../ckpt/720p/shake-3M_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_ckpt300e.pth
