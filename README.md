# Open-World Reinforcement Learning over Long Short-Term Imagination

## Getting Strated
LS-Imagine is implemented and tested on Ubuntu 20.04 with python == 3.9:

1) Create an environment
```bash
conda create -n ls_imagine python=3.9
conda activate ls_imagine
```
2) Install dependencies
```bash
pip install -r requirements.txt
```

## MineDojo

Training command of *harvest log in plains* task:  

```bash
MINEDOJO_HEADLESS=1 python expr.py --configs minedojo --task minedojo_DV3_mineclip_harvest_log_in_plains --logdir ./logdir/DV3/log --seed 10
```

## Acknowledgement
The codes refer to the implemention of [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). Thanks for the authors！



