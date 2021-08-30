#!/bin/bash
#SBATCH  --output=/scratch_net/biwinura/celleaume/contrastive_learning/sbatch_tutorial/sbatch_log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=40G


source /home/celleaume/.bashrc 

source /scratch_net/biwinura/celleaume/conda/bin/activate metastases

cd /scratch_net/biwinura/celleaume/contrastive_learning/sbatch_tutorial

python test.py

