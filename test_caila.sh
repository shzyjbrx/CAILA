#!/bin/bash
#SBATCH --job-name=caila-test
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00  # 测试过程很快，不需要申请 24 小时
#SBATCH --output=logs/test/utzappos/caila-%j.out
#SBATCH --error=logs/test/utzappos/caila-%j.err

# --- 环境加载 ---
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6

source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

CONFIG="configs/caila/zappos_large.yml" 

# --- 测试参数配置 ---
# 【重要】请将这里的路径，修改为你训练时实际生成的日志文件夹路径！
# 程序会自动去这个文件夹下寻找 `ckpt_best_auc.t7` 权重文件和配置信息。
LOGPATH="logs/caila/zappos/release-large"

echo ">>> Start Testing CAILA | Logpath: ${LOGPATH} <<<"

# 运行测试代码 (测试通常不需要分布式启动，直接用普通的 python 即可)
python test.py --config ${CONFIG} --logpath ${LOGPATH}