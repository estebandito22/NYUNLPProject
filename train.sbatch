#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000
#SBATCH -t48:00:00
#SBATCH --output=slurm_train_nmt_%j.out

## Make sure we have access to HPC-managed libraries.

# Run.
PYTHONPATH=$PYTHONPATH:. python train_enc_dec.py \
    --word_embdim 512 \
    --enc_hidden_dim 512 \
    --dec_hidden_dim 512 \
    --enc_num_layers 1 \
    --dec_num_layers 2 \
    --enc_dropout 0.0 \
    --dec_dropout 0.0 \
    --batch_size 64 \
    --lr 0.001 \
    --weight_decay 0 \
    --source_lang vi \
    --num_epochs 20 \
    --save_dir outputs_vi
