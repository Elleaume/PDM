#!/bin/bash
#SBATCH  --output=/scratch_net/biwinura/celleaume/contrastive_learning/sbatch_log/%j.out
#SBATCH  --time=24:00:00
#SBATCH  --cpus-per-task=5
#SBATCH  --gres=gpu:1
#SBATCH  --mem=50G
#SBATCH  -J results


source /home/celleaume/.bashrc 

source /scratch_net/biwinura/celleaume/conda/bin/activate metastases

cd /scratch_net/biwinura/celleaume/contrastive_learning

python3 results_test_scores_USZ.py --config configs/encoder_pretrain.json

