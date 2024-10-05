from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod
import numpy as np

class MobsWrapper(Wrapper, ABC):
    def __init__(self, env, mobs, rel_positions):
        super().__init__(env)
        self.wrapper_name = "MobsWrapper"
        self.mobs = mobs
        self.rel_positions = rel_positions

    def reset(self):
        print("==========Now is resetting from MobsWrapper!==========")

        tmp = super().reset()
        self.env.spawn_mobs(self.mobs, self.rel_positions)

        print("==========Resetting from MobsWrapper is done!==========")
        return tmp