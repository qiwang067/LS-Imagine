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
        self.HUD = HUD
        self.mask = cv2.imread('envs/tasks/base/HUD_mask_bigger.png', cv2.IMREAD_GRAYSCALE)
        if save_dir:
            self.log_dir = save_dir
        else:
            self.log_dir = os.path.join(log_dir, "screenshot")

    def reset(self):
        print("==========Now is resetting from ScreenshotWrapper!==========")

        obs = super().reset()
        
        # if self.reset_flag:
        #     self.screenshot(obs, "reset")

        self.ff = True
        self.freq = 0
        
        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        print("==========Resetting from ScreenshotWrapper is done!==========")
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
                
                
        if not self.HUD:
            img_without_HUD = self.remove_HUD(obs)
            obs['rgb'] = img_without_HUD

        self.freq += 1

        if self.reset_flag and self.ff:
            self.screenshot(obs, "reset")
            self.ff = False

        # if self.step_flag and self.freq % self.save_freq == 0:
        if self.step_flag and random.random() < (1.0 / self.save_freq):
            self.screenshot(obs, "step")

        return obs, reward, done, info
    


    def screenshot(self, obs, type):
        #
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        #
        img = self._get_curr_frame(obs)
        #
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        #
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  #
        filename = f"{type}_{current_time}.png"
        filepath = os.path.join(self.log_dir, filename)
        
        #
        img.save(filepath)
        
        # print(f"Screenshot saved at {filepath}")

    def remove_HUD(self, obs):
        img = self._get_curr_frame(obs) # [160, 256, 3]
        result = cv2.inpaint(img, self.mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

        result = result.transpose((2, 0, 1)) # [3, 160, 256]

        return result
        # pass
    
    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()