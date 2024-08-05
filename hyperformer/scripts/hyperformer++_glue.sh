#!/bin/bash
#SBATCH --job-name=hl_glue           # Job name
#SBATCH --output=logs/hyperloader_glue.%A_%a.log   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:4                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8                 # To enable the use of up to 8 GPUs

python3 -m torch.distributed.launch --nproc_per_node=4 ./finetune_t5_trainer.py configs/hyperformer++_glue.json 