#!/bin/bash
#SBATCH --job-name=habitdrive_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/habitdrive_eval/logs/smoke_%j.out
#SBATCH --error=/path/to/habitdrive_eval/logs/smoke_%j.err

set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate habitdrive
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /path/to/habitdrive_eval/code
echo "Started at $(date --iso-8601=seconds) on $(hostname)"
python smoke.py --gpu 0
echo "Completed at $(date --iso-8601=seconds)"
