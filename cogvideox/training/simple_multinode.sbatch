#!/bin/bash
#SBATCH --job-name=XYZ
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=28
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --output=output/%j-%N.out
#SBATCH --error=output/%j-%N.err
#SBATCH --qos=scavenger
#SBATCH --signal=B:USR2@300


set -x -e


source ~/.bashrc
conda activate gencam
cd /datasets/sai/gencam/cogvideox/training

echo "START TIME: $(date)"

# Training setup
GPUS_PER_NODE=4
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID 
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


#CMD="accelerate_test.py"
CMD="train_controlnet.py --config /datasets/sai/gencam/cogvideox/training/configs/gopro_train.yaml"
LAUNCHER="accelerate launch \
  --multi_gpu \
  --gpu_ids 0,1,2,3 \
  --num_processes $WORLD_SIZE \
  --num_machines $NNODES \
  --machine_rank \$SLURM_PROCID \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --rdzv_backend=c10d \
  --max_restarts 0 \
  --tee 3 \
"

# # NOT SURE THE FOLLOWING ENV VARS IS STRICTLY NEEDED (PROBABLY NOT)
# export CUDA_HOME=/usr/local/cuda-11.6
# export LD_PRELOAD=$CUDA_HOME/lib/libnccl.so
# export LD_LIBRARY_PATH=$CUDA_HOME/efa/lib:$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1 | tee logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.txt

echo "END TIME: $(date)"