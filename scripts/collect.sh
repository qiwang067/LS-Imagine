#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./script/collect.sh <task_name>"
  exit 1
fi

export MINEDOJO_HEADLESS=1
python collect_rollouts.py \
    --configs minedojo \
    --task minedojo_$1 \
    --logdir ./affordance_map/finetune_unet
