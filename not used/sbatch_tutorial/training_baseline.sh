#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH -J baseline

conda activate metastases

ipython3 ../train.py -- --config ../configs/encoder_pretrain.json
