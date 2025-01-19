#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: ./scripts/affordance.sh <task_name> <prompt>"
  exit 1
fi

TASK_NAME=$1
PROMPT=$2

python affordance_map/generate_dataset_for_finetuning.py \
    variant=attn \
    ckpt.path="weights/mineclip_attn.pth" \
    finetune_task="minedojo_${TASK_NAME}" \
    prompt="${PROMPT}"
