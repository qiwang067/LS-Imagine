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

3. Install Java: JDK `1.8.0_171`. Then install the [MineDojo](https://github.com/MineDojo/MineDojo) environment and [MineCLIP](https://github.com/MineDojo/MineCLIP) following their official documents.
4. Download the MineCLIP weight [here](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view?usp=sharing) and place them at `weights/mineclip_attn.pth`.
5. Download the Multimodal U-Net weight [here](https://drive.google.com/file/d/1Ylhw-MkT1UIUX5EyOosNmF09bWSlEjSf/view?usp=sharing), rename it to `swin_unet_checkpoint.pth`, place it at `finetune_unet/finetune_checkpoints/harvest_wool_in_plains`

## MineDojo

Training command of *Shear sheep* task:  

```bash
MINEDOJO_HEADLESS=1 python expr.py --configs minedojo --task minedojo_longdream_harvest_wool_in_plains --logdir ./logdir/ls_imagine/log --seed 10
```

## Acknowledgement
The codes refer to the implemention of [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). Thanks for the authors！



