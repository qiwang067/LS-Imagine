import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datetime import datetime

def closest_odd(x):
    x_rounded = round(x)
    if x_rounded % 2 != 0:
        return x_rounded
    else:
        return x_rounded - 1

class Concentration():

    def __init__(self, clip, device, img, prompt, ker_size, strides):
        self.clip = clip
        self.device = device
        self.img = img
        self.prompt = prompt
        self.ker_size = ker_size
        self.strides = strides
        self.final_mask = np.zeros_like(self.img)
        self.mask_list = []
        self.frame_list = []
        self.frame_region = []
        self.clip_score = 0
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.video_feats = None
        self.text_feats = None

    def generate_frames(self, rand=False):
        img_height, img_width, _ = self.img.shape

        x1 = 0
        y1 = 0
        x2 = img_width
        y2 = img_height
        
        left_up_width_step = (self.x1 - x1) / 15
        left_up_height_step = (self.y1 - y1) / 15
        right_down_width_step = (x2 - self.x2) / 15
        right_down_height_step = (y2 - self.y2) / 15

        frame_region_tmp = []
        frame_list_tmp = []

        for i in range(16):
            left_up_x = x1 + left_up_width_step * i
            left_up_y = y1 + left_up_height_step * i
            right_down_x = x2 - right_down_width_step * i
            right_down_y = y2 - right_down_height_step * i

            if rand:
                threshold = np.random.randint(100, 225) 
            else:
                threshold = 150

            if (i * i) > threshold:
                frame_region_tmp.append([int(left_up_x), int(left_up_y), int(right_down_x), int(right_down_y)])
            
            cropped_img = self.img[int(left_up_y):int(right_down_y), int(left_up_x):int(right_down_x)]
            resized_img = cv2.resize(cropped_img, (256, 160))
            frame_list_tmp.append(resized_img)

        self.frame_list.append(frame_list_tmp)
        self.frame_region.append(frame_region_tmp)

    @torch.no_grad() 
    def frames_process(self):
        self.frame_list = [torch.stack([torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() for frame in frame_list_tmp]) for frame_list_tmp in self.frame_list]
        self.frame_list = torch.stack(self.frame_list)

        self.frame_list = self.frame_list.to(self.device)
        if not isinstance(self.frame_list, torch.Tensor) or len(self.frame_list.shape) != 5:
            raise ValueError("frame_list must be a 5D Tensor with shape [frames, channels, height, width].")
        
        image_features = self.clip.forward_image_features(self.frame_list)
        self.video_feats = self.clip.forward_video_features(image_features)

    @torch.no_grad() 
    def prompt_process(self):
        if not isinstance(self.prompt, str):
            raise ValueError("prompt must be a string.")

        self.text_feats = self.clip.encode_text([self.prompt]) 

    @torch.no_grad() 
    def compute_clip_score(self):
        if self.video_feats is None or self.text_feats is None:
            raise ValueError("video_feats and text_feats must be computed before computing clip score.")

        logits_per_video, logits_per_text = self.clip.forward_reward_head(
            self.video_feats, text_tokens=self.text_feats
        )

        self.clip_score = logits_per_video.squeeze()

    def generate_mask(self):
        if self.clip_score is None:
            raise ValueError("clip_score must be computed before generating mask.")

        for regions, clip_score in zip(self.frame_region, self.clip_score):
            for region in regions:
                mask = np.zeros(self.img.shape[:2], dtype=np.float32)
                mask[region[1]:region[3], region[0]:region[2]] = clip_score.item()
                self.mask_list.append(mask)


    def generate_masks(self, rand=False):
        box_width = int(self.img.shape[1] * self.ker_size)
        box_height = int(self.img.shape[0] * self.ker_size)

        stride_x = (self.img.shape[1] - box_width) // (self.strides-1)
        stride_y = (self.img.shape[0] - box_height) // (self.strides-1)

        for y_step in range(self.strides):
            for x_step in range(self.strides):
                self.x1 = x_step * stride_x
                self.y1 = y_step * stride_y
                self.x2 = self.x1 + box_width
                self.y2 = self.y1 + box_height

                self.x2 = min(self.x2, self.img.shape[1])
                self.y2 = min(self.y2, self.img.shape[0])

                if x_step == self.strides - 1:
                    self.x2 = self.img.shape[1]

                if y_step == self.strides - 1:
                    self.y2 = self.img.shape[0]

                self.generate_frames(rand=rand)

        self.frames_process()
        self.prompt_process()
        self.compute_clip_score()
        self.generate_mask()

    
    def merge_masks(self):
        if not self.mask_list:
            raise ValueError("mask_list is empty. Generate masks before merging them.")

        coverage_count = np.full_like(self.mask_list[0], 0.1, dtype=np.float32)
        sum_of_masks = np.zeros_like(self.mask_list[0], dtype=np.float32)

        for mask in self.mask_list:
            sum_of_masks += mask
            coverage_count += (mask > 0).astype(np.float32)

        self.final_mask = sum_of_masks / coverage_count
        box_width = int(self.img.shape[1] * self.ker_size)
        box_height = int(self.img.shape[0] * self.ker_size)
        guassian_x = closest_odd(box_width)
        guassian_y = closest_odd(box_height)
        self.final_mask = cv2.GaussianBlur(self.final_mask, (guassian_x, guassian_y), 0)

        return self.final_mask

    def show_masks(self, normalize=False, save=True, save_dir=None):
        if self.final_mask is None:
            raise ValueError("final_mask is not computed. Merge masks before showing them.")

        if save:
            if not save_dir:
                output_dir = os.path.join("output", datetime.now().strftime("%Y%m%d-%H%M%S"))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            else:
                output_dir = save_dir

        plt.imshow(self.img)

        if not normalize:
            norm = Normalize(vmin=self.final_mask.min(), vmax=self.final_mask.max())
            plt.imshow(self.final_mask, cmap='jet', alpha=0.5, norm=norm)

        else:
            clipped_mask = self.final_mask

            def normalize_value(value):
                return 1 / (1 + np.exp(-1.2 * (value - 21.8)))
            
            normalized_mask = normalize_value(clipped_mask)
            norm = Normalize(vmin=0, vmax=1)
            plt.imshow(normalized_mask, cmap='jet', alpha=0.5, norm=norm)
        
        plt.axis('off')

        if save:
            counter = 0
            filename = f"{self.prompt}_{self.ker_size}_{self.strides}_{counter}.png"
            full_path = os.path.join(output_dir, filename)

            while os.path.exists(full_path):
                counter += 1
                filename = f"single_mask_{counter}.png"
                full_path = os.path.join(output_dir, filename)

            plt.savefig(full_path)
            plt.show()

        return normalized_mask, norm