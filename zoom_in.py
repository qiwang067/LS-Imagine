from typing import List, Tuple, Dict
from omegaconf import OmegaConf
from mineclip import MineCLIP
from abc import ABC, abstractstaticmethod

from envs.tasks.base.utils import *

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from envs.tasks.base.config_setting_mc import setting_config

from timm.models.layers import trunc_normal_
import math
import cv2
import torchvision.transforms as T

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from datetime import datetime

class ValueMap(ABC):
    def __init__(self, batch_size=32, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.text_feature = None
        self.prompts = None
        self.batch_size = batch_size
        self.resolution = (160, 256)

        self.mean = 150.0
        self.std = 30.0

        self.stride = 2

        self.device = kwargs.pop("device", "cuda")
        self.model = None
        self.unet = None
        self.gaussian, self.gaussian_nean = self._generate_gaussian_distribution(height=self.resolution[0], width=self.resolution[1], peak=1.0, sigma_x=128.0, sigma_y=80.0)
        # self.gaussian_mean = np.mean(self.gaussian)
        self.concentration_config = setting_config
        self.masks = None
        # self.transformer = self.concentration_config.test_transformer
        
        self.ker_size = 0.15
        self.strides = 9
        self.zoom_in_frames = []
        self.video_feats = None
        self.zoom_in_score = None

        self.index = 0

        self._load_mineclip(ckpt, kwargs)
        self._load_unet()

    def _load_mineclip(self, ckpt, config):
        '''
        '''
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = torch.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()

    def _load_unet(self):
        # config = setting_config
        model_cfg = self.concentration_config.model_config
        if self.concentration_config.network == 'egeunet':
            self.unet = EGEUNet(num_classes=model_cfg['num_classes'], 
                            input_channels=model_cfg['input_channels'], 
                            c_list=model_cfg['c_list'], 
                            bridge=model_cfg['bridge'],
                            gt_ds=model_cfg['gt_ds'],
                            )
        elif self.concentration_config.network == 'cmunet':
            self.unet = CMUNet(num_classes=model_cfg['num_classes'], 
                            input_channels=model_cfg['input_channels'], 
                            c_list=model_cfg['c_list'], 
                            bridge=model_cfg['bridge'],
                            gt_ds=model_cfg['gt_ds'],
                            )
        else: raise Exception('network in not right!')
        self.unet = self.unet.cuda()

        resume_model = os.path.join(self.concentration_config.model_checkpoint_path, 'latest.pth')
        checkpoint = torch.load(resume_model, map_location='cpu')
        self.unet.load_state_dict(checkpoint['model_state_dict'])

        #
        self.unet.eval()

    def _get_text_feats(
        self,
        prompts: str
    ) -> torch.Tensor:

        if self.prompts is not None and self.text_feature is not None and self.prompts == prompts:
            return self.text_feature # shape: [P, 512]
        
        else:
            self.prompts = prompts
            self.text_feature = self.model.encode_text(prompts)
            assert len(self.text_feature.shape) == 2 and self.text_feature.shape[0] == len(prompts), "Found shape {}".format(self.text_feature.shape)
            return self.text_feature # shape: [P, 512]


    def _generate_mask(self,
                       image_batch: torch.Tensor, # [B, H, W, C]
                       prompts: List[str]):
        # curr_frame = self._get_curr_frame(obs) # shape: [160, 256, 3]
        # save_image(curr_frame)

        # Check whether the batch size is equal to self.batch_size
        # if image_batch.shape[0] != self.batch_size:
            # raise ValueError(f"image_batch.shape[0] ({image_batch.shape[0]}) != self.batch_size ({self.batch_size})")
        
        # resized_frame, _ = self.concentration_config.test_transformer((curr_frame, np.random.rand(256, 256, 1))) # shape: [3, 160, 256]

        image_batch = image_batch.permute(0, 3, 1, 2) # shape: [B, C, H, W]

        img_normalized = (image_batch - self.mean) / self.std
        min_vals = torch.amin(img_normalized, dim=(1, 2, 3), keepdim=True)
        max_vals = torch.amax(img_normalized, dim=(1, 2, 3), keepdim=True)

        #
        img_normalized = (img_normalized - min_vals) / (max_vals - min_vals) * 255.
        
        img_resized = F.interpolate(img_normalized, size=(160, 256), mode='bilinear', align_corners=False) # shape: [B, 3, 160, 256]


        # img = resized_frame.unsqueeze(0) # shape: [1, 3, 160, 256]
        # curr_frame = curr_frame.unsqueeze(0) # shape: [1, 3, 160, 256]
        # curr_frame_0 = curr_frame.clone()

        
        with torch.no_grad():
            text_feats = self._get_text_feats(prompts).repeat(img_resized.shape[0], 1).cuda(non_blocking=True) # shape: [B, 512]
            img = img_resized.cuda(non_blocking=True).type(torch.cuda.FloatTensor) # shape: [B, 3, 160, 256]
            # img = img.expand(text_feats.shape[0], -1, -1, -1).cuda(non_blocking=True).type(torch.cuda.FloatTensor) # shape: [B, 3, 160, 256]

            # print("img.shape: ", img.shape)
            # print("text_feats.shape: ", text_feats.shape)

            _, out = self.unet(img, text_feats) # out.shape: [B, 1, 160, 256]

            out = out.squeeze(1) # shape: [B, 160, 256]
            
            # out_np = out.squeeze(1).cpu().detach().numpy() # shape: [B, 160, 256]

        self.index += 1
        # print(self.index)
        # save_image(curr_frame, self.index)
        # save_mask(out_np, self.index)
        
        

        return out
        # return out.cpu().detach().numpy()
    
    def _generate_gaussian_distribution(self, height=160, width=256, peak=1.0, sigma_x=128.0, sigma_y=80.0):
        x = np.linspace(-width // 2, width // 2, width)
        y = np.linspace(-height // 2, height // 2, height)
        x, y = np.meshgrid(x, y)

        #
        gaussian = peak * np.exp(-((x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2)))) # shape: [160, 256]
        mean = np.mean(gaussian)
        #
        gaussian = torch.from_numpy(gaussian).type(torch.cuda.FloatTensor).unsqueeze(0) # shape: [1, 160, 256]

        return gaussian, mean

    def save_image(self, image_tensor):
        #
        first_image_tensor = image_tensor[0]

        #
        first_image_tensor = torch.clamp(first_image_tensor, min=0, max=255).to(torch.uint8)

        #
        #
        if first_image_tensor.shape[0] == 3:
            first_image_tensor = first_image_tensor.permute(1, 2, 0)

        #
        image = Image.fromarray(first_image_tensor.numpy())

        #
        save_path = "output_img"
        os.makedirs(save_path, exist_ok=True)

        #
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = f"{current_time}.png"

        #
        full_save_path = os.path.join(save_path, file_name)

        #
        image.save(full_save_path)



    def get_reward(
            self,
            image_batch: torch.Tensor, # [B, H, W, C]
            prompts: List[str],
    ):
        # self.save_image(image_batch)
        
        self.masks = self._generate_mask(image_batch, prompts) # shape: [B, 160, 256]
        #
        # self.mask = np.max(masks, axis=0) * 255.0 # shape: [160, 256]
        # height, width = self.resolution
        # gaussian_distribution = self._generate_gaussian_distribution(height, width, peak=1.0, sigma_x=width/2, sigma_y=height/2)
        # score = 0
        # for mask in masks:
        #     # score += 0.5
        #     score += (np.mean(mask * self.gaussian)/self.gaussian_mean)

        # gaussian_expanded = self.gaussian.unsqueeze(0) # shape: [1, 160, 256]
        elementwise_mul = torch.mul(self.masks, self.gaussian) # shape: [B, 160, 256]
        score = torch.mean(elementwise_mul, dim=[1, 2]) # shape: [B]
        score = score.unsqueeze(1) # shape: [B, 1]

        return score

    def generate_zoom_in_frame(self, 
                               images, # [B, 64, 64, 3]
                               ):
        B, H, W, C = images.shape
        #
        masks_resized = F.interpolate(self.masks.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1) # shape: [B, 64, 64]
        
        #
        # max_locs = torch.argmax(masks_resized.view(B, -1), dim=1)
        # max_coords = torch.stack((max_locs % W, max_locs // W), dim=1)  # [B, 2] (x, y)

        #
        means = masks_resized.mean(dim=(1, 2), keepdim=True) # shape: [B, 1, 1]
        greater_than_mean = masks_resized > means # shape: [B, 64, 64]
        proportions = greater_than_mean.float().mean(dim=(1, 2)) # shape: [B]
        sqrt_proportions = torch.sqrt(proportions) # shape: [B]
        
        
        #
        window_width = (sqrt_proportions * W).int()   #
        window_height = (sqrt_proportions * H).int()  #

        #
        # window_size = torch.stack([window_width, window_height], dim=1) # shape: [B, 2]
        # window_size = torch.tensor([int(0.6 * W), int(0.6 * H)])

        #
        zoomed_images = torch.zeros_like(images)
        # print("zoomed_images.shape:", zoomed_images.shape)
        #
        # masks_resized_unsqueeze = masks_resized.unsqueeze(1)
        
        #
        for i in range(B):
            #
            cur_window_height, cur_window_width = window_height[i], window_width[i]
            kernel = torch.ones((1, 1, cur_window_height, cur_window_width), device=images.device)

            #
            #
            mask = masks_resized[i].unsqueeze(0).unsqueeze(0)
            conv_result = F.conv2d(mask, kernel, stride=self.stride)

            #
            best_value, best_idx = torch.max(conv_result.view(-1), 0)
            best_y, best_x = divmod(best_idx.item(), conv_result.shape[-1])

            #
            left_top = torch.tensor([best_x, best_y], device=images.device) * self.stride
            right_bottom = left_top + torch.tensor([cur_window_width, cur_window_height], device=images.device)

            # print("left_top:", left_top, "right_bottom:", right_bottom)

            #
            window = images[i, left_top[1]:right_bottom[1], left_top[0]:right_bottom[0], :]
            window_with_batch = window.permute(2, 0, 1).unsqueeze(0)  # shape: [1, 3, window_height, window_width]

            #
            zoomed_image = F.interpolate(window_with_batch, size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

            #
            zoomed_images[i] = zoomed_image.squeeze(0)

        # self.save_image(zoomed_images)

        # print("zoomed_images.shape:", zoomed_images.shape)
        return zoomed_images # shape: [B, 64, 64, 3]


    def get_slide_window(self):
        #
        # save_mask(self.mask)

        max_index = np.argmax(self.mask.flatten())
        max_y, max_x = np.unravel_index(max_index, self.mask.shape)
        window_h, window_w = [dim * self.ker_size for dim in self.resolution]

        # print("====================", self.index)
        # print("max_y, max_x:", max_y, max_x)

        #
        left_top_y = min(max(0, max_y - window_h // 2), self.resolution[0] - window_h)
        left_top_x = min(max(0, max_x - window_w // 2), self.resolution[1] - window_w)
        right_bottom_y = left_top_y + window_h
        right_bottom_x = left_top_x + window_w

        return left_top_y, left_top_x, right_bottom_y, right_bottom_x

    def generate_zoom_in_frames(self, obs, left_top_y, left_top_x, right_bottom_y, right_bottom_x, device='cuda'):
        #
        zoom_in_frames_tensor = torch.empty((16, 3, self.resolution[0], self.resolution[1]), dtype=torch.float32, device=device)
        
        #
        left_up_width_step = (left_top_x - 0) / 15.0
        left_up_height_step = (left_top_y - 0) / 15.0
        right_down_width_step = (self.resolution[1] - right_bottom_x) / 15.0
        right_down_height_step = (self.resolution[0] - right_bottom_y) / 15.0

        #
        curr_frame_tensor = torch.from_numpy(self._get_curr_frame(obs)).to(device).float().permute(2, 0, 1)

        #
        for i in range(16):
            left_top_x_ = int(left_up_width_step * i)
            left_top_y_ = int(left_up_height_step * i)
            right_bottom_x_ = self.resolution[1] - int(right_down_width_step * i)
            right_bottom_y_ = self.resolution[0] - int(right_down_height_step * i)

            #
            zoom_in_frame_tensor = T.functional.crop(curr_frame_tensor, left_top_y_, left_top_x_, right_bottom_y_ - left_top_y_, right_bottom_x_ - left_top_x_)
            zoom_in_frame_tensor = T.functional.resize(zoom_in_frame_tensor, self.resolution, interpolation=T.InterpolationMode.BILINEAR)
            
            #
            zoom_in_frames_tensor[i].copy_(zoom_in_frame_tensor)

        self.zoom_in_frames = zoom_in_frames_tensor
        
    @torch.no_grad()
    def frames_process(self, obs):
        self.generate_zoom_in_frames(obs, *self.get_slide_window())

        # self.zoom_in_frames = torch.stack([torch.from_numpy(np.transpose(frame, (2, 0, 1))).float() for frame in self.zoom_in_frames]) # shape: [16, 3, 160, 256]
        # self.zoom_in_frames = [torch.from_numpy(np.transpose(frame, (2, 0, 1))) for frame in self.zoom_in_frames] # shape: [16, 3, 160, 256]
        
        #
        # self.zoom_in_frames = [torch.from_numpy(np.transpose(frame, (2, 0, 1))) for frame in self.zoom_in_frames]
        
        # self.zoom_in_frames = torch.rand(16, 3, 160, 256)

        self.zoom_in_frames = self.zoom_in_frames.to(self.device)
        self.zoom_in_frames = self.zoom_in_frames.unsqueeze(0) # shape: [1, 16, 3, 160, 256]
        # print("self.zoom_in_frames.shape:", self.zoom_in_frames.shape)

        image_features = self.model.forward_image_features(self.zoom_in_frames) 
        self.video_feats = self.model.forward_video_features(image_features)

    @torch.no_grad()
    def compute_zoom_in_score(self, prompts):
        if self.video_feats is None:
            self.frames_process()
        logits_per_video, logits_per_text = self.model.forward_reward_head(
            self.video_feats, text_tokens=self._get_text_feats(prompts)
        )

        self.zoom_in_score = logits_per_text.squeeze()


def custom_sigmoid(x, slope=1.2):
    return 1 / (1 + torch.exp(-slope * x))

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding, 
                      stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, k_size=3, d_list=[1,2,5,7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[0]-1))//2, 
                      dilation=d_list[0], groups=group_size + 1)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[1]-1))//2, 
                      dilation=d_list[1], groups=group_size + 1)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[2]-1))//2, 
                      dilation=d_list[2], groups=group_size + 1)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size+1, data_format='channels_first'),
            nn.Conv2d(group_size + 1, group_size + 1, kernel_size=3, stride=1, 
                      padding=(k_size+(k_size-1)*(d_list[3]-1))//2, 
                      dilation=d_list[3], groups=group_size + 1)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4, dim_xl, 1)
        )
    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode ='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)
        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0,x1,x2,x3), dim=1)
        x = self.tail_conv(x)
        return x


class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    def __init__(self, dim_in, dim_out, x=8, y=8):
        super().__init__()
        
        c_dim_in = dim_in//4
        k_size=3
        pad=(k_size-1) // 2
        
        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in), nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
                nn.Conv2d(c_dim_in, c_dim_in, 1),
                nn.GELU(),
                nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )
        
        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        
        self.ldw = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
                nn.GELU(),
                nn.Conv2d(dim_in, dim_out, 1),
        )
        
    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        B, C, H, W = x1.size()
        #----------xy----------#
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4],mode='bilinear', align_corners=True))
        #----------zx----------#
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(F.interpolate(params_zx, size=x2.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        #----------zy----------#
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(F.interpolate(params_zy, size=x3.shape[2:4],mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        #----------dw----------#
        x4 = self.dw(x4)
        #----------concat----------#
        x = torch.cat([x1,x2,x3,x4],dim=1)
        #----------ldw----------#
        x = self.norm2(x)
        x = self.ldw(x)
        return x

'''
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N, _, H, W = values.shape
        value_len, key_len = H * W, H * W

        #
        values = values.reshape(N, self.heads, self.head_dim, value_len).transpose(2, 3)
        keys = keys.reshape(N, self.heads, self.head_dim, key_len).transpose(2, 3)

        #
        #
        queries = query.expand(-1, -1, H, W)
        queries = queries.reshape(N, self.heads, self.head_dim, value_len).transpose(2, 3)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, self.heads * self.head_dim, value_len
        ).transpose(1, 2)

        #
        out = self.fc_out(out).reshape(N, -1, H, W)

        return out


class TextImageAttention(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, heads):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(image_feature_dim, heads)
        self.text_projection = nn.Linear(text_feature_dim, image_feature_dim)
        #
        #

    def forward(self, image_features, text_features):
        text_features_projected = self.text_projection(text_features).unsqueeze(2).unsqueeze(3)
        attention_output = self.multi_head_attention(image_features, image_features, text_features_projected)
        
        #
        combined_features = attention_output + image_features
        
        return combined_features
'''
 
class EGEUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, text_feature):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out # b, c0, H/2, W/2

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out # b, c1, H/4, W/4 

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out # b, c2, H/8, W/8
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out # b, c3, H/16, W/16
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out # b, c4, H/32, W/32
        
        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = out
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) # b, c4, H/32, W/32
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        '''
        if self.gt_ds:
            return (torch.sigmoid(gt_pre5), torch.sigmoid(gt_pre4), torch.sigmoid(gt_pre3), torch.sigmoid(gt_pre2), torch.sigmoid(gt_pre1)), torch.sigmoid(out0)
        else:
            return torch.sigmoid(out0)  
        '''  
        
        if self.gt_ds:
            return (custom_sigmoid(gt_pre5), custom_sigmoid(gt_pre4), custom_sigmoid(gt_pre3), custom_sigmoid(gt_pre2), custom_sigmoid(gt_pre1)), custom_sigmoid(out0)
        else:
            return custom_sigmoid(out0)

'''
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q: query, shape (batch_size, n_heads, d_k, height, width)
        k: key, shape (batch_size, n_heads, d_k, height, width)
        v: value, shape (batch_size, n_heads, d_v, height, width)
        mask: optional attention mask, shape (batch_size, 1, height, width)
        """
        bsz, n_heads, d_k, h, w = q.size()

        # Compute attention scores using scaled dot-product
        print("q shape: ", q.shape)
        attn = torch.einsum('bnqhw,bnkhw->bnhwk', q, k) / self.temperature  # shape: (batch_size, n_heads, height, width, height, width)
        print("attn shape: ", attn.shape)

        # Apply the mask if provided
        if mask is not None:
            mask = mask.view(bsz, 1, 1, h, w).expand(-1, n_heads, -1, -1, -1)  # shape: (batch_size, n_heads, 1, height, width)
            attn = attn.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention probabilities
        attn = self.softmax(attn)  # shape: (batch_size, n_heads, height, width, height, width)

        # Apply dropout to attention probabilities
        attn = self.dropout(attn)

        # Compute the weighted sum of values using attention probabilities
        output = torch.einsum('bnhwk,bnvhw->bnvhw', attn, v)  # shape: (batch_size, n_heads, d_v, height, width)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Conv2d(d_model, n_heads * d_k, kernel_size=1, bias=False)
        self.w_ks = nn.Conv2d(d_model, n_heads * d_k, kernel_size=1, bias=False)
        self.w_vs = nn.Conv2d(d_model, n_heads * d_v, kernel_size=1, bias=False)
        self.fc = nn.Conv2d(n_heads * d_v, d_model, kernel_size=1, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

    def forward(self, q, k, v, mask=None):
        """
        q: query, shape (batch_size, channels, height, width)
        k: key, shape (batch_size, channels, height, width)
        v: value, shape (batch_size, channels, height, width)
        mask: optional attention mask, shape (batch_size, 1, height, width)
        """
        bsz, _, h, w = q.size()
        d_model = self.n_heads * self.d_k

        # Project q, k, v using different learned projection layers
        q = self.w_qs(q).view(bsz, self.n_heads, self.d_k, h, w)  # shape: (batch_size, n_heads, d_k, height, width)
        k = self.w_ks(k).view(bsz, self.n_heads, self.d_k, h, w)  # shape: (batch_size, n_heads, d_k, height, width)
        v = self.w_vs(v).view(bsz, self.n_heads, self.d_v, h, w)  # shape: (batch_size, n_heads, d_v, height, width)

        # Compute attention using the attention module
        output, attn = self.attention(q, k, v, mask=mask)  # output shape: (batch_size, n_heads, d_v, height, width), attn shape: (batch_size, n_heads, height, width, height, width)

        # Reshape and apply a final linear transformation
        output = output.view(bsz, self.n_heads * self.d_v, h, w)  # shape: (batch_size, n_heads * d_v, height, width)
        output = self.fc(output)  # shape: (batch_size, d_model, height, width)

        return output, attn
    
'''

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        '''
        q, k, v shape: (batch_size, n_head, height * width, d_k)
        '''
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # shape: (batch_size, n_head, height * width, height * width)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1)) # shape: (batch_size, n_head, height * width, height * width)
        output = torch.matmul(attn, v) # shape: (batch_size, n_head, height * width, d_v)

        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        '''
        q: query(text_features), q.shape: (batch_size, height * width, image_feature_dim)
        k: key(image_features), k.shape: (batch_size, height * width, image_feature_dim)
        v: value(image_features), v.shape: (batch_size, height * width, image_feature_dim)
        '''

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k) # shape: (batch_size, height * width, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k) # shape: (batch_size, height * width, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v) # shape: (batch_size, height * width, n_head, d_v)

        # Transpose for attention dot product: (batch_size, n_head, height * width, d_k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1) # For head axis broadcasting.
        
        output, attn = self.attention(q, k, v, mask=mask) # shape: (batch_size, n_head, height * width, height * width)

        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1) # shape: (batch_size, height * width, n_head * d_v)
        output = self.dropout(self.fc(output)) # shape: (batch_size, height * width, d_model)
        output = self.layer_norm(output) # shape: (batch_size, height * width, d_model)
        
        return output, attn
        

class FusionLayer(nn.Module):
    def __init__(self, channel):
        super(FusionLayer, self).__init__()
        self.fuse = nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, residual):
        #
        out = torch.cat([x, residual], dim=1)
        #
        out = self.fuse(out)
        #
        out = self.norm(out)
        out = self.relu(out)
        return out

class ConvFusion(nn.Module):
    def __init__(self, channels):
        super(ConvFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
        # self.norm = nn.LayerNorm(channels)

    def forward(self, feature1, feature2):
        #
        combined_features = torch.cat((feature1, feature2), dim=1)  #
        fused_features = self.conv(combined_features)
        #
        #
        #
        # fused_features = self.norm(fused_features)
        #
        return fused_features

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super(AttentionFusion, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature1, feature2):
        combined_features = torch.cat((feature1, feature2), dim=1)
        attention_weights = self.attention(combined_features)
        fused_features = feature1 * attention_weights + feature2 * (1 - attention_weights)
        return fused_features
    
class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feature1, feature2):
        combined_features = torch.cat((feature1, feature2), dim=1)
        gate = self.gate(combined_features)
        fused_features = feature1 * gate + feature2 * (1 - gate)
        return fused_features

class TextImageAttention(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, n_heads):
        super().__init__()
        # Assuming that the text_feature_dim can be divided by n_heads
        self.text_projection = nn.Linear(text_feature_dim, image_feature_dim)
        self.multi_head_attention = MultiHeadAttention(n_heads, image_feature_dim, image_feature_dim, image_feature_dim)
        self.fusion_layer = AttentionFusion(image_feature_dim)

    def forward(self, image_features, text_features):
        # image_features shape: (batch_size, channels, height, width)
        # text_features shape: (batch_size, text_feature_dim)

        height, width = image_features.shape[2], image_features.shape[3]

        residual = image_features
        
        image_features = torch.reshape(image_features, (-1, image_features.shape[1], image_features.shape[2] * image_features.shape[3])) # shape: (batch_size, channels, height * width)
        image_features = image_features.permute(0, 2, 1) # shape: (batch_size, height * width, image_feature_dim)

        text_features = self.text_projection(text_features) # shape: (batch_size, image_feature_dim)
        text_features = text_features.unsqueeze(1).expand(-1, image_features.shape[1], -1) # shape: (batch_size, height * width, image_feature_dim)

        attention, _ = self.multi_head_attention(q=text_features, k=image_features, v=image_features) # shape: (batch_size, height * width, image_feature_dim)
        # print(attention.shape)

        attention = attention.permute(0, 2, 1) # shape: (batch_size, image_feature_dim, height * width
        # print(attention.shape)
        attention = attention.reshape(attention.shape[0], attention.shape[1], height, width) # shape: (batch_size, image_feature_dim, height, width)

        combined_features = self.fusion_layer(attention, residual) # shape: (batch_size, image_feature_dim, height, width)

        return combined_features

    
class CMUNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, text_fiture_dim=512, heads=8, c_list=[8,16,24,32,48,64], bridge=True, gt_ds=True):
        super().__init__()

        self.bridge = bridge
        self.gt_ds = gt_ds
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[2], c_list[3]),
        )
        self.encoder5 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[4]),
        )
        self.encoder6 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[5]),
        )

        
        # self.text_image_attention_1 = TextImageAttention(c_list[0], text_fiture_dim, heads)
        # self.text_image_attention_2 = TextImageAttention(c_list[1], text_fiture_dim, heads)
        # self.text_image_attention_3 = TextImageAttention(c_list[2], text_fiture_dim, heads)
        self.text_image_attention_4 = TextImageAttention(c_list[3], text_fiture_dim, heads)
        self.text_image_attention_5 = TextImageAttention(c_list[4], text_fiture_dim, heads)
        self.text_image_attention_6 = TextImageAttention(c_list[5], text_fiture_dim, heads)
        
        if bridge: 
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0])
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1])
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2])
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3])
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4])
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], 1, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], 1, 1))
            print('gt deep supervision was used')
        
        self.decoder1 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[5], c_list[4]),
        ) 
        self.decoder2 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[4], c_list[3]),
        ) 
        self.decoder3 = nn.Sequential(
            Grouped_multi_axis_Hadamard_Product_Attention(c_list[3], c_list[2]),
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # def forward(self, x, text_features):
    def forward(self, x, text_features):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        # t1 = self.text_image_attention_1(out, text_features) # b, c0, H/2, W/2
        t1 = out

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        # t2 = self.text_image_attention_2(out, text_features) # b, c1, H/4, W/4 
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        # t3 = self.text_image_attention_3(out, text_features) # b, c2, H/8, W/8
        t3 = out

        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = self.text_image_attention_4(out, text_features) # b, c3, H/16, W/16
        # t4 = out

        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = self.text_image_attention_5(out, text_features) # b, c4, H/32, W/32
        # t5 = out

        out = F.gelu(self.encoder6(out)) # b, c5, H/32, W/32
        t6 = self.text_image_attention_6(out, text_features)
        # t6 = out

        out5 = F.gelu(self.dbn1(self.decoder1(t6))) # b, c4, H/32, W/32
        if self.gt_ds: 
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode ='bilinear', align_corners=True)
        else: t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5) # b, c4, H/32, W/32
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c3, H/16, W/16
        if self.gt_ds: 
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode ='bilinear', align_corners=True)
        else:t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4) # b, c3, H/16, W/16
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c2, H/8, W/8
        if self.gt_ds: 
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode ='bilinear', align_corners=True)
        else: t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3) # b, c2, H/8, W/8
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c1, H/4, W/4
        if self.gt_ds: 
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode ='bilinear', align_corners=True)
        else: t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2) # b, c1, H/4, W/4 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) # b, c0, H/2, W/2
        if self.gt_ds: 
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode ='bilinear', align_corners=True)
        else: t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1) # b, c0, H/2, W/2
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) # b, num_class, H, W
        
        if self.gt_ds:
            return (custom_sigmoid(gt_pre5), custom_sigmoid(gt_pre4), custom_sigmoid(gt_pre3), custom_sigmoid(gt_pre2), custom_sigmoid(gt_pre1)), custom_sigmoid(out0)
        else:
            return custom_sigmoid(out0)




