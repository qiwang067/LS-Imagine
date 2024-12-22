from typing import Dict
from gym import Wrapper
import numpy as np
from abc import ABC, abstractstaticmethod

class TerminalWrapper(Wrapper, ABC):
    def __init__(self, env, max_steps: int = 500, on_death=True, all: Dict = dict(), any: Dict = dict(), stagger_max_steps=False):
        super().__init__(env)
        self.wrapper_name = "TerminalWrapper"
        self.max_steps = max_steps
        self.on_death = on_death
        self.all_conditions = all
        self.any_conditions = any
        self.t = 0
        self.curr_max_steps = self.max_steps
        self.stagger_max_steps = stagger_max_steps

    def reset(self):
        self.t = 0
        if self.stagger_max_steps:
            self.curr_max_steps = np.random.randint((self.max_steps*3)//4, self.max_steps+1)
        else:
            self.curr_max_steps = self.max_steps

        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.t += 1
        info['real_done'] = False

        if self.t >= self.curr_max_steps:
            done = True
            if self.t > self.curr_max_steps:
                info["success"] = False

        if self.on_death:
            if self._check_condition("death", {}, obs):
                done = True
                info['real_done'] = True

        if len(self.all_conditions) > 0:
            if all(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.all_conditions.items()
            ):
                done = True
                info['real_done'] = True

        if len(self.any_conditions) > 0:
            if any(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.any_conditions.items()
            ):
                done = True
                info['real_done'] = True

        return obs, reward, done, info

    def _check_condition(self, condition_type, condition_info, obs):
        if condition_type == "item":
            return self._check_item_condition(condition_info, obs)
        elif condition_type == "blocks":
            return self._check_blocks_condition(condition_info, obs)
        elif condition_type == "death":
            return self._check_death_condition(condition_info, obs)
        else:
            raise NotImplementedError("{} terminal condition not implemented".format(condition_type))

    @abstractstaticmethod
    def _check_item_condition(condition_info, obs):
        raise NotImplementedError()

    @abstractstaticmethod
    def _check_blocks_condition(condition_info, obs):
        raise NotImplementedError()

    @abstractstaticmethod
    def _check_death_condition(condition_info, obs):
        raise NotImplementedError()