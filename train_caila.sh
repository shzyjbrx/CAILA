#!/bin/bash
#SBATCH --job-name=caila-train
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/train/cgqa/caila-%j.out
#SBATCH --error=logs/train/cgqa/caila-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

# --- 1. 数据集软链接配置 ---
# CAILA 默认从 ./all_data 读取数据。这里创建软链接指向您的实际数据目录。
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data"

# # 直接将你 Base/data 下的源文件夹，强制链接为代码需要的 ut-zap50k 名字
# ln -snf ${DATA_ROOT}/ut-zappos ./all_data/ut-zap50k
# 注意：您的路径是 ut-zappos，但原版 CAILA 配置文件中写的是 ut-zap50k。
# 如果运行 Zappos 数据集，请将您的文件夹重命名为 ut-zap50k，或在 configs 对应的 yml 中修改 data_dir 为 ut-zappos。

# --- 2. 预训练权重配置 ---
# CAILA 需要从 clip_ckpts 文件夹加载 HuggingFace 格式的权重
# mkdir -p ./clip_ckpts
# 请确保您的权重转换为了 HuggingFace 格式并放在这里，例如：
# ln -snf /path/to/huggingface/clip-vit-large-patch14.pth ./clip_ckpts/clip-vit-large-patch14.pth

# --- 3. 训练参数配置 ---
# 请根据需要选择对应的 config 文件。这里以 MIT-States 的 Large 模型为例
CONFIG="configs/caila/cgqa_large.yml"

# 如果使用单卡，N_GPU 设为 1
N_GPU=4

echo ">>> Start Training CAILA | Config: ${CONFIG} <<<"

# CAILA 原代码要求使用 torchrun 启动
python -m torch.distributed.run --nproc_per_node=${N_GPU} train.py --config ${CONFIG}