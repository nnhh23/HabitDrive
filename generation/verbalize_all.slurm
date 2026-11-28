#!/bin/bash
#SBATCH --job-name=habitdrive_verbalize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/path/to/habitdrive_full/logs/verbalize_%j.out
#SBATCH --error=/path/to/habitdrive_full/logs/verbalize_%j.err

set -u
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate habitdrive
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd /path/to/habitdrive_full/code
mkdir -p ../data/history ../logs

echo "Started at $(date --iso-8601=seconds) on $(hostname)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

for i in 0 1 2 3; do
  python verbalize.py --gpu "$i" --shard "$i" --num-shards 4 \
    --input ../data/structured_history.json --output-dir ../data/history \
    --batch-size 8 > ../logs/verbalize_gpu$i.log 2>&1 &
done
wait

echo "Completed at $(date --iso-8601=seconds)"
ls ../data/history/ | wc -l
