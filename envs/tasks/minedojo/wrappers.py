import torch as th
import numpy as np
import cv2
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
    
class LLaVABERTScoreReward(BertscoreReward):
    @staticmethod
    def _get_curr_frame(obs, image_size):
        #
        rgb_array = obs["rgb"]
        
        #
        if rgb_array.shape != (3, image_size[0], image_size[1]):
            raise ValueError("The shape of the RGB data does not match the expected image size.")
        
        #
        rgb_image = np.transpose(rgb_array, (1, 2, 0))
        
        #
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        return bgr_image
    
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


class MinedojoTechTreeWrapper(TechTreeWrapper):
    def _get_inventory(self, obs):
        """"""
        inventory = {
            name: sum(
                int(obs["inventory"]["quantity"][i]) 
                for i, n in enumerate(obs["inventory"]["name"]) if name_match(name, n)
            )
            for name in self._get_all_items()
        }
        inventory.pop("dirt", None)
        inventory.pop("air", None)
        inventory["log"] += inventory.pop("log2", 0)
        return inventory

    def _get_all_items(self):
        """"""
        return list(set(ALL_ITEMS) - set(["dirt", "log2", "air"]))

    def _get_craftables(self):
        """"""
        return ALL_CRAFTING_TABLE_ITEMS + ALL_SMELTING_ITEMS

    def _get_noop_action(self):
        """"""
        return {
            handler.to_string(): handler.space.no_op() 
            for handler in self.unwrapped._sim_spec.actionables
        }

    def _get_craft_action(self, item, crafting_table=False, furnace=False, valid=None):
        """"""
        if valid is None:
            valid = self._get_all_items()
        action = {
            handler.to_string(): handler.space.no_op() 
            for handler in self.unwrapped._sim_spec.actionables
        }
        if (item in ALL_PERSONAL_CRAFTING_ITEMS or crafting_table and item in ALL_CRAFTING_TABLE_ITEMS) and \
                all(n in valid and self.curr_inventory[n] >= q for n, q in CRAFTING_RECIPES_BY_OUTPUT[item][0]["ingredients"].items()) or \
                furnace and item in ALL_SMELTING_ITEMS and \
                all(n in valid and self.curr_inventory[n] >= q for n, q in SMELTING_RECIPES_BY_OUTPUT[item][0]["ingredients"].items()):
            action["craft"] = item
            action["craft_with_table"] = item
            action["smelt"] = item
        else:
            action = None
        return action

    def _equip_item(self, obs, action, item):
        """"""
        if obs is None:
            return action
        hotbar_items = [x.replace(" ", "_") for x in obs["inventory"]["name"][:9].tolist()]
        if item in hotbar_items:
            hotbar_names = ["hotbar." + str(x) for x in range(1, 10)]
            for handler in self.unwrapped._sim_spec.actionables:
                if handler.to_string() in ["drop", "swap_slot", "pickItem"] + hotbar_names:
                    action[handler.to_string()] = handler.space.no_op()
            action[hotbar_names[hotbar_items.index(item)]] = np.array(1)
        return action


class MinedojoRewardWrapper(RewardWrapper):
    @staticmethod
    def _get_item_count(obs, item):
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) if name_match(item, name))

class MinedojoMobsWrapper(MobsWrapper):
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
        """"""
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        """"""
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
        """"""
        return sum(quantity for name, quantity in zip(obs["inventory"]["name"], obs["inventory"]["quantity"]) 
                   if name_match(condition_info["type"], name)) >= condition_info["quantity"]

    @staticmethod
    def _check_blocks_condition(condition_info, obs):
        """"""
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

class MinedojoDV3Wrapper(DV3Wrapper):
    pass

""""""