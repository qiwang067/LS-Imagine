import torch as th
import numpy as np
from minedojo.sim.wrappers.fast_reset import FastResetWrapper
from minedojo.sim.mc_meta.mc import ALL_ITEMS, ALL_PERSONAL_CRAFTING_ITEMS, ALL_CRAFTING_TABLE_ITEMS, ALL_SMELTING_ITEMS,\
    CRAFTING_RECIPES_BY_OUTPUT, SMELTING_RECIPES_BY_OUTPUT

from envs.tasks.base import *


def name_match(target_name, obs_name):
    return target_name.replace(" ", "_") == obs_name.replace(" ", "_")

# Fast reset wrapper saves time but doesn't replace blocks
# Occasionally doing a hard reset should prevent state shift
class MinedojoSemifastResetWrapper(FastResetWrapper):

    def __init__(self, *args, reset_freq=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_freq = reset_freq
        self.reset_count = 0

    def reset(self):
        if self.reset_count < self.reset_freq:
            self.reset_count += 1
            return super().reset()
        else:
            self.reset_count = 0
            return self.env.reset()


class MinedojoClipReward(ClipReward):
    @staticmethod
    def _get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        return th.from_numpy(curr_frame)

    @staticmethod
    def get_resolution():
        return (160, 256)
    
class MinedojoConcentrationReward(ConcentrationReward):
    @staticmethod
    def get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        curr_frame = curr_frame.transpose((1, 2, 0))
        return curr_frame # shape: (160, 256, 3)
    
    @staticmethod
    def get_resolution():
        return (160, 256)


class MinedojoRewardWrapper(RewardWrapper):
    @staticmethod
    def _get_item_count(obs, item):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) if name_match(item, name))


class MinedojoScreenshotWrapper(ScreenshotWrapper):
    @staticmethod
    def _get_curr_frame(obs):
        curr_frame = obs["rgb"].copy()
        curr_frame = curr_frame.transpose((1, 2, 0))
        return curr_frame # shape: (160, 256, 3)
    
    @staticmethod
    def get_resolution():
        return (160, 256)
    

class MinedojoSuccessWrapper(SuccessWrapper):
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False


class MinedojoTerminalWrapper(TerminalWrapper):
    @staticmethod
    def _check_item_condition(condition_info, obs):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        target = np.array(condition_info)
        voxels = obs["voxels"]["block_name"].transpose(1,0,2)
        for y in range(voxels.shape[0] - target.shape[0]):
            for x in range(voxels.shape[1] - target.shape[1]):
                for z in range(voxels.shape[2] - target.shape[2]):
                    if np.all(voxels[y:y+target.shape[0],
                                     x:x+target.shape[1],
                                     z:z+target.shape[2]] == target):
                        return True
        return False

    @staticmethod
    def _check_death_condition(condition_info, obs):
        return obs["life_stats"]["life"].item() == 0


class MinedojoLSImagineWrapper(LSImagineWrapper):
    pass
