#!/bin/bash
#SBATCH  --output=/scratch_net/biwinura/celleaume/contrastive_learning/sbatch_log/%j.out
#SBATCH  --time=24:00:00
#SBATCH  --cpus-per-task=5
#SBATCH  --gres=gpu:1
#SBATCH  --mem=60G
#SBATCH  -J train_segmentation


source /home/celleaume/.bashrc 

source /scratch_net/biwinura/celleaume/conda/bin/activate metastases

cd /scratch_net/biwinura/celleaume/contrastive_learning

python3 training_2v_5.py --config configs/encoder_pretrain.json

