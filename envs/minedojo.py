from envs.tasks import make, get_specs
import numpy as np

import gym

def make_env(task, *args, **kwargs):
    return make(task, **kwargs)