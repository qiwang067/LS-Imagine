<h1 align="center"> Open-World Reinforcement Learning over Long Short-Term Imagination </h1>
<p align="center">
    Jiajian Li*
    ·
    Qi Wang*
    ·
    Yunbo Wang
    ·
    Xin Jin
    ·
    Yang Li
    ·
    Wenjun Zeng
    ·
    Xiaokang Yang
  </p>

<h3 align="center"> <a href="https://arxiv.org/pdf/2410.03618" target="_blank"> arXiv </a> &nbsp;&nbsp; | &nbsp;&nbsp; <a href="https://qiwang067.github.io/ls-imagine" target="_blank"> Website </a> &nbsp;&nbsp; </h3>
  <div align="center"></div>

<p align="center">
<img src="assets/overview.png" alt="Teaser image" />
</p>

<p style="text-align:justify">
 Training visual reinforcement learning agents in a high-dimensional open world presents significant challenges. While various model-based methods have improved sample efficiency by learning interactive world models, these agents tend to be “short-sighted”, as they are typically trained on short snippets of imagined experiences. We argue that the primary obstacle in open-world decision-making is improving the efficiency of off-policy exploration across an extensive state space. In this paper, we present LS-Imagine, which extends the imagination horizon within a limited number of state transition steps, enabling the agent to explore behaviors that potentially lead to promising long-term feedback. The foundation of our approach is to build a <i>short-term world model</i>. To achieve this, we simulate goal-conditioned jumpy state transitions and compute corresponding affordance maps by zooming in on specific areas within single images. This facilitates the integration of direct long-term values into behavior learning. Our method demonstrates significant improvements over state-of-the-art techniques in MineDojo. 
</p>

<!-- # Open-World Reinforcement Learning over Long Short-Term Imagination
#### Open-World Reinforcement Learning over Long Short-Term Imagination

Jiajian Li*, Qi Wang*, Yunbo Wang, Xin Jin, Yang Li, Wenjun Zeng, Xiaokang Yang

[[arXiv]](https://arxiv.org/pdf/2410.03618)  [[Project Page]](https://qiwang067.github.io/ls-imagine) -->

## Getting Strated
LS-Imagine is implemented and tested on Ubuntu 20.04 with python == 3.9:

1) Create an environment
```bash
conda create -n ls_imagine python=3.9
conda activate ls_imagine
```

2. Install Java: JDK `1.8.0_171`. Then install the [MineDojo](https://github.com/MineDojo/MineDojo) environment and [MineCLIP](https://github.com/MineDojo/MineCLIP) following their official documents.

3) Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the MineCLIP weight [here](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view?usp=sharing) and place them at `weights/mineclip_attn.pth`.
5. Download the Multimodal U-Net weight [here](https://drive.google.com/file/d/1Ylhw-MkT1UIUX5EyOosNmF09bWSlEjSf/view?usp=sharing), rename it to `swin_unet_checkpoint.pth`, place it at `finetune_unet/finetune_checkpoints/harvest_wool_in_plains`

## MineDojo

Training command of *Shear sheep* task:  

```bash
MINEDOJO_HEADLESS=1 python expr.py --configs minedojo --task minedojo_harvest_wool_in_plains --logdir ./logdir
```

## Citation
```bib
@article{li2024open,
  title={Open-World Reinforcement Learning over Long Short-Term Imagination}, 
  author={Jiajian Li and Qi Wang and Yunbo Wang and Xin Jin and Yang Li and Wenjun Zeng and Xiaokang Yang},
  journal={arXiv preprint arXiv:2410.03618},
  year={2024}
}
```

## Acknowledgement
The codes refer to the implemention of [dreamerv3-torch](https://github.com/NM512/dreamerv3-torch). Thanks for the authors！


