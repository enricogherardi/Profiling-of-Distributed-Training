#!/bin/bash
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --partition=m100_usr_prod
#SBATCH -A PHD_aimageth
#SBATCH -t 02:00:00
#SBATCH --output=log/prova_out.txt
#SBATCH --error=log/prova_err.txt
#SBATCH --nodelist=r244n17,r244n18

MAST_ADDR=10.39.44.17
EPOCHS=2
WORKERS=16
BATCH=32

srun -N1 -n1 -w r244n17 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=0 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=1 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=2 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=3 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=4 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=5 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=6 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
srun -N1 -n1 -w r244n18 --gres=gpu:4 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=8 --node_rank=7 --master_addr="$MAST_ADDR" --master_port=35000 --use_env train.py --epochs "$EPOCHS" -j "$WORKERS" --batch-size "$BATCH" &
wait