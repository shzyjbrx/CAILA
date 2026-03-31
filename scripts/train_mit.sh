#!/bin/bash
#SBATCH --job-name=p2-mit
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# 1. 修改日志输出路径，归类到 mit 文件夹下
#SBATCH --output=logs/point2/mit/train/p2-%j.out
#SBATCH --error=logs/point2/mit/train/p2-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

# --- 1. 数据集软链接配置 ---
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data"

# --- 2. 预训练权重配置 ---

# --- 3. 训练参数配置 ---
# 2. 将配置文件指向 MIT-States 的 ViT-L/14 配置
CONFIG="configs/caila/mit_large.yml"

# 如果使用单卡，N_GPU 设为 1；当前申请了 4 卡，保持 4
N_GPU=4

# 层级消融实验参数 (对应 ViT-L/14 的 24 层)
# 23 = 仅最后1层
# 18 = 浅深层 (最后6层)
# 12 = 中深层 (最后12层)
# 0  = 全层 (全部24层)
ADAPTER_START_LAYER=0
BATCH_SIZE=16

echo ">>> Start Training Dual-Branch Baseline | Dataset: MIT-States | Config: ${CONFIG} | Start Layer: ${ADAPTER_START_LAYER} <<<"

export OMP_NUM_THREADS=1

# 4. 在启动命令中追加 --adapter_start_layer 参数覆盖 yml 中的默认配置
python -m torch.distributed.run --nproc_per_node=${N_GPU} train.py \
    --config ${CONFIG} \
    --adapter_start_layer ${ADAPTER_START_LAYER} \
    --batch_size ${BATCH_SIZE}