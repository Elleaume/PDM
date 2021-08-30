export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
srun --time 02:00:00 --partition=gpu.debug --gres=gpu:1 --pty bash -i