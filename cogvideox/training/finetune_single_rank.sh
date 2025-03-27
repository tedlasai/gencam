#!/bin/bash

export MODEL_PATH="/datasets/sai/gencam/cogvideox/CogVideoX-2b/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_controlnet.py \
  --tracker_name "cogvideox-controlnet" \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --validation_prompt "" \
  --validation_video "../resources/car.mp4:::../resources/ship.mp4" \
  --validation_prompt_separator ::: \
  --num_inference_steps 28 \
  --num_validation_videos 1 \
  --validation_steps 100 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir "cogvideox-controlnet" \
  --height 512 \
  --width 512 \
  --fps 8 \
  --max_num_frames 25 \
  --video_root_dir "set-path-to-video-directory" \
  --csv_path "set-path-to-csv-file" \
  --stride_min 1 \
  --stride_max 3 \
  --hflip_p 0.5 \
  --downscale_coef 8 \
  --init_from_transformer \
  --train_batch_size 1 \
  --dataloader_num_workers 0 \
  --num_train_epochs 100 \
  --checkpointing_steps 100 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 
  # --report_to wandb
  # --pretrained_controlnet_path "cogvideox-controlnet-2b/checkpoint-2000.pt" \
    