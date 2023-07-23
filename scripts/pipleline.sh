# Bash scripts pipeline training and testing for 7 videos in UVG dataset

### 3M-720p
# ReadySetGo (AvgPSNR: 36.54)
# echo "Training ReadySetGo 720x1080, model size = 3M, for 300 epochs"
# python scripts/train.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
# echo "Completed training ReadySetGo 720x1080, model size = 3M, training 300 epochs"
# echo "Evaluating PSNR & Inference video for ReadySetGo 720x1080, model size = 3M from ckpte300.pth"
# python scripts/test.py --config config/uvg-720p/3M/ready_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/Ready-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth
# echo "Completed caculate PSNR & inference video for ReadySetGo 720x1080, model size = 3M from ckpte299.pth"

# Yatch Ride
echo "Training YachtRide 720x1080, model size = 3M, for 300 epochs"
python scripts/train.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml
echo "Completed training YachtRide 720x1080, model size = 3M, training 300 epochs"

echo "Evaluating PSNR & Inference video for YachtRide 720x1080, model size = 3M from ckpte300.pth"
python scripts/test.py --config config/uvg-720p/3M/yacht_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e.yaml --resume saved/models/Yacht-HD_vmaev2-adaptive3d-nervb3d_b2xf4-cosinelr-20k_300e/e299.pth
echo "Completed caculate PSNR & inference video for YachtRide 720x1080, model size = 3M from ckpte299.pth"
