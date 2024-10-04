from gym import Wrapper
import gym.spaces as spaces
import numpy as np
from abc import ABC, abstractmethod
import threading
import cv2
from collections import OrderedDict

import copy

BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.])),
    "turn_down": dict(camera=np.array([10.0, 0.])),
    "turn_left": dict(camera=np.array([0., -10.0])),
    "turn_right": dict(camera=np.array([0., 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    # "place_dirt": dict(place="dirt"),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    'camera': np.array([0., 0.]), 
    'smelt': 'none', 
    'craft': 'none', 
    'craft_with_table': 'none', 
    'forward': np.array(0), 
    'back': np.array(0), 
    'left': np.array(0), 
    'right': np.array(0), 
    'jump': np.array(0), 
    'sneak': np.array(0), 
    'sprint': np.array(0), 
    'use': np.array(0), 
    'attack': np.array(0), 
    'drop': 0, 
    'swap_slot': OrderedDict([('source_slot', 0), ('target_slot', 0)]), 
    'pickItem': 0, 
    'hotbar.1': 0, 
    'hotbar.2': 0, 
    'hotbar.3': 0, 
    'hotbar.4': 0, 
    'hotbar.5': 0, 
    'hotbar.6': 0, 
    'hotbar.7': 0, 
    'hotbar.8': 0, 
    'hotbar.9': 0,
}


class DV3Wrapper(Wrapper, ABC):
    def __init__(self, env, repeat=1, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70), zoom_in=False, fusion=False):
        super().__init__(env)
        self.wrapper_name = "DV3Wrapper"

        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        #
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())
        self._zoom_in = zoom_in
        self._fusion = fusion
        print(f"==========DV3Wrapper zoom_in: {self._zoom_in}==========")
        print(f"==========DV3Wrapper fusion: {self._fusion}==========")

        if fusion:
            self.observation_space = spaces.Dict(
                {
                    'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                    # 'zoomed_image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                    'heatmap': spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8),
                    # 'heatmap_on_zoomed': spaces.Box(-np.inf, np.inf, shape=(64, 64, 1), dtype=np.float32),
                    'jump': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_calculated': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'reward_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'intrinsic': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'intrinsic_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'score': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'score_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'jumping_steps': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'accumulated_reward': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                }
            )

        else:
            self.observation_space = spaces.Dict(
                {
                    'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                    # 'zoomed_image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                    'jump': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_calculated': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                    'reward_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'intrinsic': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'intrinsic_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'score': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'score_on_zoomed': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'jumping_steps': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                    'accumulated_reward': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                }
            )

        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True

        # self._noop_action = 
        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        # with threading.Lock():
        obs = self.env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def step(self, action):
        action = copy.deepcopy(self._action_values[action])
        action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        # obs["is_terminal"] = bool(info.get("is_terminal", done))
        if self._zoom_in:
            obs["is_terminal"] = bool(info.get("is_terminal", info["real_done"]))
        else:
            obs["is_terminal"] = bool(info.get("is_terminal", done))

        obs = self._obs(obs)
        
        # print("=========dv3 wrapper=========")
        # print(f"obs['zoom_in_threshold']: {obs['zoom_in_threshold']}")
        # print(f"obs['zoom_in_mean']: {obs['zoom_in_mean']}")
        assert "pov" not in obs, list(obs.keys())
        return obs, reward, done, info


    
    def _obs(self, obs):
        image = obs['rgb'] # 3 * H * W
        image = image.transpose(1, 2, 0).astype(np.uint8) # H * W * 3
        image = cv2.resize(image, (64, 64)) # 64 * 64 * 3

        if 'zoomed_image' in obs:
            zoomed_image = obs['zoomed_image'] # H * W * 3
            zoomed_image = zoomed_image.astype(np.uint8) # H * W * 3
            zoomed_image = cv2.resize(zoomed_image, (64, 64)) # 64 * 64 * 3
        else:
            zoomed_image = np.zeros_like(image)

        if self._fusion:
            heatmap = cv2.resize(obs['heatmap'], (64, 64))
            heatmap_on_zoomed = cv2.resize(obs['heatmap_on_zoomed'], (64, 64))

            #
            heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
            heatmap_on_zoomed = np.clip(heatmap_on_zoomed * 255, 0, 255).astype(np.uint8)

            obs = {
                'image': image,
                # 'zoomed_image': zoomed_image,
                'heatmap': heatmap,
                # 'heatmap_on_zoomed': heatmap_on_zoomed,
                'jump': obs['jump'] if 'jump' in obs else False,
                'is_zoomed': obs['is_zoomed'] if 'is_zoomed' in obs else False,
                'is_calculated': obs['is_calculated'] if 'is_calculated' in obs else False,
                'is_first': obs['is_first'],
                'is_last': obs['is_last'],
                'is_terminal': obs['is_terminal'],
                # 'concentration_score': obs['concentration_score'] if 'concentration_score' in obs else 0.0,
                # 'zoom_in_prob': obs['zoom_in_prob'] if 'zoom_in_prob' in obs else 0.0,
                # 'concentration_score_on_zoomed': obs['concentration_score_on_zoomed'] if 'concentration_score_on_zoomed' in obs else 0.0,
                # 'zoom_in_prob_on_zoomed': obs['zoom_in_prob_on_zoomed'] if 'zoom_in_prob_on_zoomed' in obs else 0.0,
                'reward_on_zoomed': obs['reward_on_zoomed'] if 'reward_on_zoomed' in obs else 0.0,
                # 'zoom_in_threshold': obs['zoom_in_threshold'],
                # 'zoom_in_mean': obs['zoom_in_mean'],
                # 'zoom_in_std_dev': obs['zoom_in_std_dev'],
                'intrinsic': obs['intrinsic'] if 'intrinsic' in obs else 0.0,
                'intrinsic_on_zoomed': obs['intrinsic_on_zoomed'] if 'intrinsic_on_zoomed' in obs else 0.0,
                'score': obs['score'] if 'score' in obs else 0.0,
                'score_on_zoomed': obs['score_on_zoomed'] if 'score_on_zoomed' in obs else 0.0,
                'jumping_steps': obs['jumping_steps'] if 'jumping_steps' in obs else 1000.0,
                'accumulated_reward': obs['accumulated_reward'] if 'accumulated_reward' in obs else 1000.0,
            }

            if self._zoom_in:
                if obs["is_zoomed"]:
                    obs["zoomed_image"] = zoomed_image
                    obs["heatmap_on_zoomed"] = heatmap_on_zoomed
                else:
                    obs["zoomed_image"] = None
                    obs["heatmap_on_zoomed"] = None

        else:
            obs = {
                'image': image,
                # 'zoomed_image': zoomed_image,
                'jump': obs['jump'] if 'jump' in obs else False,
                'is_zoomed': obs['is_zoomed'] if 'is_zoomed' in obs else False,
                'is_calculated': obs['is_calculated'] if 'is_calculated' in obs else False,
                'is_first': obs['is_first'],
                'is_last': obs['is_last'],
                'is_terminal': obs['is_terminal'],
                # 'concentration_score': obs['concentration_score'] if 'concentration_score' in obs else 0.0,
                # 'zoom_in_prob': obs['zoom_in_prob'] if 'zoom_in_prob' in obs else 0.0,
                # 'concentration_score_on_zoomed': obs['concentration_score_on_zoomed'] if 'concentration_score_on_zoomed' in obs else 0.0,
                # 'zoom_in_prob_on_zoomed': obs['zoom_in_prob_on_zoomed'] if 'zoom_in_prob_on_zoomed' in obs else 0.0,
                'reward_on_zoomed': obs['reward_on_zoomed'] if 'reward_on_zoomed' in obs else 0.0,
                # 'zoom_in_threshold': obs['zoom_in_threshold'],
                # 'zoom_in_mean': obs['zoom_in_mean'],
                # 'zoom_in_std_dev': obs['zoom_in_std_dev'],
                'intrinsic': obs['intrinsic'] if 'intrinsic' in obs else 0.0,
                'intrinsic_on_zoomed': obs['intrinsic_on_zoomed'] if 'intrinsic_on_zoomed' in obs else 0.0,
                'score': obs['score'] if 'score' in obs else 0.0,
                'score_on_zoomed': obs['score_on_zoomed'] if 'score_on_zoomed' in obs else 0.0,
                'jumping_steps': obs['jumping_steps'] if 'jumping_steps' in obs else 1000.0,
                'accumulated_reward': obs['accumulated_reward'] if 'accumulated_reward' in obs else 1000.0,
            }

            if self._zoom_in:
                if obs["is_zoomed"]:
                    obs["zoomed_image"] = zoomed_image
                else:
                    obs["zoomed_image"] = None

        for key, value in obs.items():
            if key in self.observation_space:
                space = self.observation_space[key]
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                assert (key, value, value.dtype, value.shape, space)
        return obs

    def _action(self, action):
        # print("action_01:", action)
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        # print("action_02:", action)
        # print(action["camera"][0])
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        # print("self.pitch:", self._pitch)
        return action


    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions

        