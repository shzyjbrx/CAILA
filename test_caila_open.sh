#!/bin/bash
#SBATCH --job-name=caila-open-test
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/test/utzappos/caila-open-%j.out
#SBATCH --error=logs/test/utzappos/caila-open-%j.err

# --- 1. 环境加载 ---
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
module unload cudnn
module load cudnn/8.4.0.27_cuda11.x

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

# --- 2. 测试参数配置 ---
# 注意：请根据你当前想要测试的数据集，核对这两个路径
CONFIG="configs/caila/zappos_large.yml" 
LOGPATH="logs/caila/zappos/release-large"

echo ">>> Start OPEN WORLD Testing CAILA | Config: ${CONFIG} | Logpath: ${LOGPATH} <<<"

# --- 3. 运行测试 ---
# 【关键修改】在命令最末尾加上了 --open_world 标志
python test.py --config ${CONFIG} --logpath ${LOGPATH} --open_world