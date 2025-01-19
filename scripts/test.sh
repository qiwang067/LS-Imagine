#!/bin/bash

# 检查是否提供了必要的参数
if [ "$#" -lt 3 ]; then
  echo "Usage: ./scripts/test.sh <path_to_latest.pt> <eval_episode_num> <task_name>"
  exit 1
fi

# 参数赋值
AGENT_CHECKPOINT_DIR=$1
EVAL_EPISODE_NUM=$2
TASK_NAME=$3

# 执行 Python 脚本
export MINEDOJO_HEADLESS=1
python test.py \
    --configs minedojo \
    --task minedojo_${TASK_NAME} \
    --logdir ./logdir \
    --agent_checkpoint_dir ${AGENT_CHECKPOINT_DIR} \
    --eval_episode_num ${EVAL_EPISODE_NUM}
