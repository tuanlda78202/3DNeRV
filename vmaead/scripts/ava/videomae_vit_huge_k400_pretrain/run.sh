# Set the path to save checkpoints and logs
OUTPUT_DIR='YOUR_PATH/ava_videomae_vit_huge_k400_pretrain'
# path to pretrain model
# Google Drive Link: https://drive.google.com/file/d/1AJQR1Rsi2N1pDn9tLyJ8DQrUREiBA1bO
MODEL_PATH='YOUR_PATH_TO_PRETRAINED_MODEL/checkpoint.pth'

# batch_size can be adjusted according to number of GPUs
# this script is for 64 GPUs (8 nodes x 8 GPUs)
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port 12320 --nnodes=8 \
      --node_rank=$1 --master_addr=$2 run_class_finetuning.py \
      --model vit_huge_patch16_224 \
      --finetune ${MODEL_PATH} \
      --log_dir ${OUTPUT_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --batch_size 2 \
      --input_size 224 \
      --save_ckpt_freq 10 \
      --num_frames 16 \
      --sampling_rate 4 \
      --opt adamw \
      --lr 5e-4 \
      --layer_decay 0.8 \
      --opt_betas 0.9 0.999 \
      --weight_decay 0.05 \
      --epochs 20 \
      --data_set "ava" \
      --drop_path 0.2 \
      --val_freq 10 \
      --use_checkpoint
