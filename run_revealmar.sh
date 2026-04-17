#!/bin/bash
set -e

python main_revealmar.py \
  --model revealmar_base \
  --data_path ./data/imagenet \
  --vae_path pretrained_models/vae/kl16.ckpt \
  --resume pretrained_models/mar/mar_base \
  --evaluate \
  --num_iter 256 \
  --num_sampling_steps 100 \
  --cfg 2.9 \
  --planner_hidden_dim 128 \
  --planner_loss_weight 1.0 \
  --candidate_pool_size 8 \
  --pseudo_target_type none \
  --budget_mode soft \
  --mixed_policy_ratio 0.0
