#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: ./scripts/train.sh <task_name>"
  exit 1
fi

export MINEDOJO_HEADLESS=1
python expr.py \
    --configs minedojo \
    --task minedojo_$1 \
    --logdir ./logdir
