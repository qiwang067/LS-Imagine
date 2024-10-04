from gym import Wrapper
import torch as th
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def save_image(img, index, name='curr_frame', output_dir='output_tmp'):
    #
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imsave(os.path.join(output_dir, f"{index}_{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), img)

def save_mask(out_np, index, name='mask', output_dir='output_tmp'):
    #
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    #
    for i, mask in enumerate(out_np):
        plt.imsave(os.path.join(output_dir, f"{index}_{name}_{i}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), mask, cmap='jet', vmin=0, vmax=1)

class ConcentrationWrapper(Wrapper):
    def __init__(self, env, concentration, prompts=None, gaussian=True, fusion=False, zoom_in=True, dense_reward=0.01, clip_target=23, clip_min=21, smoothing=50, mineclip=True, mineclip_dense_reward=0.01, max_steps=1000, **kwargs):
        super().__init__(env)
        self.concentration = concentration # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.mineclip_dense_reward = mineclip_dense_reward
        self.smoothing = smoothing
        self.mineclip = mineclip
        self.gaussian = gaussian
        self.fusion = fusion
        self.zoom_in = zoom_in

        self.clip_target = clip_target
        self.clip_min = clip_min

        self.buffer = None
        self.zoom_in_buffer = None
        self.last_score = 0
        # self.last_zoom_in_score = 0

        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        self.max_steps = max_steps

    def reset(self, **kwargs):
        print("==========Now is resetting from ConcentrationWrapper!==========")

        self.buffer = None
        self.zoom_in_buffer = None
        self.last_score = 0
        # self.last_zoom_in_score = 0
        self.last_zoom_in_mineclip_score = 0
        self.last_zoom_in_gaussian_score = 0

        print("self.prompt", self.prompt)
        obs = self.env.reset(**kwargs)

        #
        if self.gaussian:
            score, zoom_in_prob, check_threshold = self.concentration.get_reward(obs, self.prompt)
        else:
            score, zoom_in_prob, check_threshold = 0.0, 0.0, 1.0
            # print("score: ", score)
        
        #
        if self.gaussian and self.zoom_in:
            zoomed_image, is_check = self.concentration.generate_zoom_in_frame()
        else:
            zoomed_image = self.concentration.get_curr_frame(obs)
            is_check = False

        #
        if is_check:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = self.concentration.compute_reward_on_zoomed_image()
        else:
            mineclip_on_zoomed, gaussian_on_zoomed, zoom_in_prob_on_zoomed, is_zoomed, jump = 0.0, 0.0, 0.0, False, False
            # zoom_in_mean, zoom_in_std_dev = self.concentration.get_zoom_in_distribution()
            # zoom_in_threshold = self.concentration.get_zoom_in_threshold()

        obs['is_zoomed'] = is_zoomed
        obs['jump'] = jump
        obs['jumping_steps'] = self.max_steps
        obs['accumulated_reward'] = 0.0
        obs['is_calculated'] = False
        # obs['zoom_in_threshold'] = zoom_in_threshold
        # obs['zoom_in_mean'] = zoom_in_mean
        # obs['zoom_in_std_dev'] = zoom_in_std_dev

        # obs['gaussian'] = 0.0
        # obs['zoom_in_prob'] = zoom_in_prob

        obs['reward_on_zoomed'] = 0.0
        obs['intrinsic_on_zoomed'] = 0.0
        obs['score_on_zoomed'] = 0.0
        obs['zoomed_image'] = zoomed_image
        # obs['zoom_in_prob_on_zoomed'] = zoom_in_prob_on_zoomed

        # self.buffer = self._insert_buffer(self.buffer, score) 
        # score = self._get_score()

        # if score > self.last_score: 
        #     self.last_score = score
        #     obs['gaussian'] = self.dense_reward * score

        if score > self.last_score:
        # if True:
            obs['intrinsic'] += self.dense_reward * score
            self.last_score = score

        obs['score'] += self.dense_reward * score

        if self.zoom_in and is_zoomed:
            if gaussian_on_zoomed > self.last_score and gaussian_on_zoomed > self.last_zoom_in_gaussian_score:
            # if True:
                obs['intrinsic_on_zoomed'] += self.dense_reward * gaussian_on_zoomed
                self.last_zoom_in_gaussian_score = gaussian_on_zoomed

            obs['score_on_zoomed'] += self.dense_reward * gaussian_on_zoomed
            
            # obs['intrinsic_on_zoomed'] = self.dense_reward * gaussian_on_zoomed
            if self.mineclip:
                if mineclip_on_zoomed > self.last_zoom_in_mineclip_score:
                # if True:
                    obs['intrinsic_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
                    self.last_zoom_in_mineclip_score = mineclip_on_zoomed

                obs['score_on_zoomed'] += self.mineclip_dense_reward * mineclip_on_zoomed
            # gaussian_on_zoomed = 0.5 * (gaussian_on_zoomed + score)

            # if gaussian_on_zoomed > self.last_score:
            #     obs['gaussian_on_zoomed'] = self.dense_reward * gaussian_on_zoomed
           
        if self.fusion:
            obs['heatmap'] = self.concentration.get_heatmap(is_zoomed=False)
            if self.zoom_in and is_zoomed:
                obs['heatmap_on_zoomed'] = self.concentration.get_heatmap(is_zoomed=True)
            else:
                obs['heatmap_on_zoomed'] = obs['heatmap']
        
        """"""
        if buffer is None:
            buffer = [logit]
        elif len(buffer) < self.smoothing:
            buffer.append(logit)
        else:
            buffer = buffer[1:] + [logit]
        return buffer        
    
    
    def _get_slide_window(self):
        #
        if self.concentration.mask is not None:
            mask = self.concentration.mask
        else:
            raise ValueError("self.concentration.mask is None, please check your code!")

                
        max_index = np.argmax(mask)
        max_row, max_col = divmod(max_index, 256)

    
    def zoom_in_reward(self):
        pass
        


    

                
