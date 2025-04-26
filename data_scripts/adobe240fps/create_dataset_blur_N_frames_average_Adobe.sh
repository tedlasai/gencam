#!/usr/bin/env bash

# we use the static build ffmpeg to de-compress the videos

python ./create_dataset_blur_N_frames_average.py \
        --dataset /home/tedlasai/genCamera/data_scripts/adobe240fps \
        --window_size 11 \
        --enable_train 1 \
        --dataset_folder ../../Adobe_240fps_dataset/Adobe_240fps_blur\
        --videos_folder  ../../Adobe_240fps_dataset/original_high_fps_videos\
        --img_width 640 \
        --img_height 352 \
        --extract_frames 1 \

