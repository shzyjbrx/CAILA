#!/bin/bash
#SBATCH --job-name=p2-cgqa
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# 1. 修改日志输出路径，归类到 cgqa 文件夹下
#SBATCH --output=logs/point2/cgqa/train/p2-%j.out
#SBATCH --error=logs/point2/cgqa/train/p2-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

# --- 1. 数据集软链接配置 ---
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data"

# --- 2. 训练参数配置 ---
# 2. 将配置文件指向 CGQA 的 ViT-L/14 配置 (请确保 configs/caila 目录下有这个文件)
CONFIG="configs/caila/cgqa_large.yml"

# 如果使用单卡，N_GPU 设为 1；当前申请了 4 卡，保持 4
N_GPU=4

# 层级消融实验参数 (对应 ViT-L/14 的 24 层)
# 0 = 全层 (全部24层)
ADAPTER_START_LAYER=0

# 自定义单卡 Batch Size (总 Batch Size = 128 * 4 = 512)
BATCH_SIZE=4

echo ">>> Start Training Dual-Branch Baseline | Dataset: CGQA | Config: ${CONFIG} | Start Layer: ${ADAPTER_START_LAYER} | Batch Size: ${BATCH_SIZE} <<<"

export OMP_NUM_THREADS=1

# 3. 启动 DDP 训练，并覆盖 yml 中的相关参数
python -m torch.distributed.run --nproc_per_node=${N_GPU} train.py \
    --config ${CONFIG} \
    --adapter_start_layer ${ADAPTER_START_LAYER} \
    --batch_size ${BATCH_SIZE}