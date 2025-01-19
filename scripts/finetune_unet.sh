#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./scripts/finetune_unet.sh <task_name>"
  exit 1
fi

python affordance_map/finetune.py \
    --root_path affordance_map/datasets/Minecraft \
    --dataset Minecraft \
    --list_dir ./affordance_map/lists/lists_Minecraft \
    --num_class 1 \
    --cfg affordance_map/configs/swin_tiny_patch4_window7_224_lite.yaml \
    --max_epochs 200 \
    --output_dir ./affordance_map/model_out \
    --base_lr 0.01 \
    --batch_size 24 \
    --finetune_task minedojo_$1
