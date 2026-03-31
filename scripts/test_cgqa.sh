#!/bin/bash
#SBATCH --job-name=p2-eval-cgqa
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/point2/cgqa/test/p2-eval-all-%j.out
#SBATCH --error=logs/point2/cgqa/test/p2-eval-all-%j.err

# --- 1. 环境加载与冲突处理 ---
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false

# --- 2. 测试参数配置 ---
CONFIG="configs/caila/cgqa_large.yml" 
# 确保您之前训练的 C-GQA 模型保存在这个路径下，如果文件夹名字有时间戳，请替换为真实的文件夹名
LOGPATH="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/CAILA/logs/caila_cgqa_release-large/cgqa_20260330_110738_caila_cgqa_release-large"

# ==============================================================================
# 阶段一：封闭世界测试 (Closed-World)
# ==============================================================================
# echo "====================================================================="
# echo ">>> [Phase 1/2] Start CLOSED WORLD Testing CAILA on C-GQA <<<"
# echo ">>> Config: ${CONFIG} | Logpath: ${LOGPATH} <<<"
# echo "====================================================================="

# python test.py --config ${CONFIG} --logpath ${LOGPATH}

# echo -e "\n\n"

# ==============================================================================
# 阶段二：开放世界测试 (Open-World)
# ==============================================================================
echo "====================================================================="
echo ">>> [Phase 2/2] Start OPEN WORLD Testing CAILA on C-GQA <<<"
echo ">>> Config: ${CONFIG} | Logpath: ${LOGPATH} <<<"
echo "====================================================================="

python test.py --config ${CONFIG} --logpath ${LOGPATH} --open_world --test_batch_size 16 --cpu_eval

echo -e "\n>>> All evaluation tasks completed successfully! <<<"