#!/bin/bash
#SBATCH --job-name=train_deblur
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1         # one Accelerate launcher per node
#SBATCH --gres=gpu:4                # 4 GPUs per node
#SBATCH --cpus-per-task=8           # e.g. 2 cores/GPU
#SBATCH --mem=64gb
#SBATCH --qos=scavenger
#SBATCH --partition=gen-part
#SBATCH --output=slurm-%j-%N.out
#SBATCH --error=slurm-%j-%N.err
#SBATCH --signal=B:USR2@300

if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <config.yaml>" && exit 1
fi
config_file=$1

# pick a single master address+port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
MASTER_PORT=$(python free-port.py $MASTER_ADDR)


# launch one bash on each node, import your env, then call Accelerate
srun --export=ALL \
     --output=slurm-%j-%N.out \
     --error=slurm-%j-%N.err \
     bash -lc '
       source ~/.bashrc
       conda activate gencam

       echo "[$(hostname)] SLURM_PROCID=$SLURM_PROCID"

       export MASTER_ADDR='"$MASTER_ADDR"'
       export MASTER_PORT='"$MASTER_PORT"'
       # total machines / world-size
       export NUM_MACHINES=$SLURM_JOB_NUM_NODES
       export NUM_PROCESSES=$(( SLURM_GPUS_ON_NODE * NUM_MACHINES ))
       export MACHINE_RANK=$SLURM_PROCID

       export NCCL_IB_DISABLE=1
       export NCCL_SOCKET_IFNAME=ens
       export PYTHONUNBUFFERED=1

       accelerate launch \
         --config_file accelerator_train_config_multinode.yaml \
         --main_process_ip   $MASTER_ADDR \
         --main_process_port $MASTER_PORT \
         --machine_rank      $MACHINE_RANK \
         --num_machines      $NUM_MACHINES \
         --num_processes     $NUM_PROCESSES \
       train.py --config '"$config_file"'
     '
