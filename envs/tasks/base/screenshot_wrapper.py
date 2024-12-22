from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod
from datetime import datetime
from PIL import Image
import numpy as np
import os
import random
import cv2

class ScreenshotWrapper(Wrapper, ABC):
    def __init__(self, env, log_dir, reset_flag=False, step_flag=False, save_freq=1, save_dir='', HUD=False, **kwargs):
        super().__init__(env)
        self.wrapper_name = "ScreenshotWrapper"
        self.reset_flag = reset_flag
        self.step_flag = step_flag
        self.ff = False
        self.save_freq = save_freq
        self.freq = 0
        self.episode = 0
        self.steps = 0
        self.HUD = HUD
        self.mask = cv2.imread('envs/tasks/base/HUD_mask.png', cv2.IMREAD_GRAYSCALE)
        if save_dir:
            self.log_dir = save_dir
        else:
            self.log_dir = os.path.join(log_dir, "screenshot")

    def reset(self):
        obs = super().reset()

        self.ff = True
        self.freq = 0
        self.episode += 1
        self.steps = 0

        if self.reset_flag and self.ff:
            self.screenshot(obs, "reset", self.episode, self.steps)
            self.ff = False
        
        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.freq += 1
        self.steps += 1

        if self.step_flag and random.random() < (1.0 / self.save_freq):
            self.screenshot(obs, "step", self.episode, self.steps)

        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        return obs, reward, done, info
    

    def screenshot(self, obs, type, episode_num, step_num):
        full_path = os.path.join(self.log_dir, f"episode_{episode_num}", "image")

        if not os.path.exists(full_path):
            os.makedirs(full_path)

        img = self._get_curr_frame(obs)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] 
        filename = f"{step_num}.png"
        filepath = os.path.join(full_path, filename)
        
        img.save(filepath)

    def remove_HUD(self, obs):
        img = self._get_curr_frame(obs)
        result = cv2.inpaint(img, self.mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
        result = result.transpose((2, 0, 1))

        return result

    
    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()