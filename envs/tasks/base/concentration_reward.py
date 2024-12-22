import os
import math
import cv2
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import copy
import torch.utils.checkpoint as checkpoint

from torch import nn
from typing import List, Dict
from omegaconf import OmegaConf
from mineclip import MineCLIP
from abc import ABC, abstractstaticmethod
from einops import rearrange
from datetime import datetime
from scipy.stats import kurtosis
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.transforms import Normalize

MC_IMAGE_MEAN = (0.3331, 0.3245, 0.3051)
MC_IMAGE_STD = (0.2439, 0.2493, 0.2873)
MC_NORMALIZER = Normalize(mean=MC_IMAGE_MEAN, std=MC_IMAGE_STD)

def normalize_numpy(image, mean, std):
    for c in range(image.shape[2]):
        image[:, :, c] = (image[:, :, c] - mean[c]) / std[c]
    return image

def save_image(img, index, name='curr_frame', output_dir='output_tmp'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imsave(os.path.join(output_dir, f"{index}_{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), img)

def save_mask(out_np, index, name='mask', output_dir='output_tmp'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, mask in enumerate(out_np):
        plt.imsave(os.path.join(output_dir, f"{index}_{name}_{i}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"), mask, cmap='jet', vmin=0, vmax=1)

def save_affordance_map(out_np, episode_num, step_num, output_dir='output_tmp'):
    save_path = os.path.join(output_dir, 'screenshot', f'episode_{episode_num}', 'map')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for i, mask in enumerate(out_np):
        plt.imsave(os.path.join(save_path, f"{step_num}.png"), mask, cmap='jet', vmin=0, vmax=1.0)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def resized_if_need(image, target_size=(256, 160)):
    if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
    return image
    
def MCResize(image, label, target_size=(256, 160)):
    image = resized_if_need(image, target_size)
    label = resized_if_need(label, target_size)
    
    return image, label

def MyToTensor(image, label):
    return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(label).permute(2, 0, 1)
    
def MNormalize(image, label):
    return normalize_numpy(image / 255.0, MC_IMAGE_MEAN, MC_IMAGE_STD), label

class Config:
    class AUG:
        AUTO_AUGMENT = "rand-m9-mstd0.5-inc1"
        COLOR_JITTER = 0.4
        CUTMIX = 1.0
        CUTMIX_MINMAX = None
        MIXUP = 0.8
        MIXUP_MODE = "batch"
        MIXUP_PROB = 1.0
        MIXUP_SWITCH_PROB = 0.5
        RECOUNT = 1
        REMODE = "pixel"
        REPROB = 0.25

    class DATA:
        BATCH_SIZE = 72
        CACHE_MODE = "part"
        DATASET = "imagenet"
        DATA_PATH = ""
        IMG_SIZE = 224
        INTERPOLATION = "bicubic"
        NUM_WORKERS = 8
        PIN_MEMORY = True
        ZIP_MODE = False

    class MODEL:
        DROP_PATH_RATE = 0.2
        DROP_RATE = 0.0
        HEADS = 8
        LABEL_SMOOTHING = 0.1
        NAME = "swin_tiny_patch4_window7_224"
        NUM_CLASSES = 1000
        PRETRAIN_CKPT = "./pretrained_ckpt/swin_tiny_patch4_window7_224.pth"
        RESUME = ""
        SWIN = type('SWIN', (), {
            'APE': False,
            'DECODER_DEPTHS': [2, 2, 2, 1],
            'DEPTHS': [2, 2, 2, 2],
            'EMBED_DIM': 96,
            'FINAL_UPSAMPLE': "expand_first",
            'IN_CHANS': 3,
            'MLP_RATIO': 4.0,
            'NUM_HEADS': [3, 6, 12, 24],
            'PATCH_NORM': True,
            'PATCH_SIZE': 4,
            'QKV_BIAS': True,
            'QK_SCALE': None,
            'WINDOW_SIZE': 7
        })
        TEXT_FEATURE_DIM = 512
        TYPE = "swin"

    class TRAIN:
        ACCUMULATION_STEPS = 0
        AUTO_RESUME = True
        BASE_LR = 0.0005
        CLIP_GRAD = 5.0
        EPOCHS = 300
        LR_SCHEDULER = type('LR_SCHEDULER', (), {
            'DECAY_EPOCHS': 30,
            'DECAY_RATE': 0.1,
            'NAME': "cosine"
        })
        MIN_LR = 5e-06
        OPTIMIZER = type('OPTIMIZER', (), {
            'BETAS': (0.9, 0.999),
            'EPS': 1e-08,
            'MOMENTUM': 0.9,
            'NAME': "adamw"
        })
        START_EPOCH = 0
        USE_CHECKPOINT = False
        WARMUP_EPOCHS = 20
        WARMUP_LR = 5e-07
        WEIGHT_DECAY = 0.05

    EVAL_MODE = True
    LOCAL_RANK = 0
    OUTPUT = ""
    PRINT_FREQ = 10
    SAVE_FREQ = 1
    SEED = 0
    TAG = "default"
    TEST = type('TEST', (), {
        'CROP': True
    })
    THROUGHPUT_MODE = False

class RandomGenerator(object):
    def __init__(self, output_height=224, output_width=224):
        self.output_height = output_height
        self.output_width = output_width

    def __call__(self, image, label):
        image, label = MCResize(image, label, target_size=(self.output_width, self.output_height))
        image, label = MNormalize(image, label)
        image, label = MyToTensor(image, label)

        return image, label

class ThresholdBuffer:
    def __init__(self):
        self.n = 0       
        self.mean = 0    
        self.M2 = 0      

    def count(self):
        return self.n
    
    def fmean(self):
        return self.mean if self.n > 0 else 0
    
    def std_dev(self):
        return math.sqrt(self.M2 / self.n) if self.n > 1 else 0
    
    def add(self, number):
        self.n += 1
        delta = number - self.mean
        self.mean += delta / self.n
        delta2 = number - self.mean
        self.M2 += delta * delta2
    
    def get_threshold(self):
        if self.n > 0:
            return self.fmean() + self.std_dev()
        else:
            return 1.0


class ConcentrationReward(ABC):
    def __init__(self, ckpt="weights/mineclip_attn.pth", unet_checkpoint_dir="envs/tasks/base/unet_checkpoint", output_dir="output_tmp", gaussian_sigma_weight=0.5, **kwargs) -> None:
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.text_feature = None
        self.prompts = None
        self.unet_checkpoint_dir = unet_checkpoint_dir
        self.output_dir = output_dir

        self.resolution = self.get_resolution() # (160, 256)
        self.u_net_resolution = 224
        self.device = kwargs.pop("device", "cuda")
        self.model = None
        self.unet = None
        self.gaussian = self._generate_gaussian_distribution(height=self.resolution[0], width=self.resolution[1], peak=1.0, sigma_x=self.resolution[1]*gaussian_sigma_weight, sigma_y=self.resolution[0]*gaussian_sigma_weight)
        self.gaussian_mean = np.mean(self.gaussian)
        self.mask = None
        self.mask_on_zoomed_image = None
        self.preprocess = RandomGenerator(output_height=self.u_net_resolution, output_width=self.u_net_resolution)
        self.best_value_on_mask = 0

        self.have_center = False

        self.ker_size = 0.15
        self.strides = 9
        self.stride = 1
        self.zoom_in_frames = []
        self.video_feats = None
        self.zoom_in_score = None

        self.index = 0

        self.unet_cfg = Config()
        self._load_mineclip(ckpt, kwargs)
        self._load_unet()

        self.gaussian_score = 0
        self.zoom_in_prob = 0
        self.num_above_threshold = 0

        self.blur_x = 51
        self.blur_y = 79 

        self.check_threshold_buffer = ThresholdBuffer()
        self.gaussian_buffer = ThresholdBuffer()

        self.check_threshold = 1

        self.curr_frame = None
        self.zoomed_frame = None

        
    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()
    
    @abstractstaticmethod
    def get_curr_frame(obs):
        raise NotImplementedError()
    
    def _load_mineclip(self, ckpt, config):
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = torch.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()

    def _load_unet(self):
        self.unet = MCUnet(self.unet_cfg, img_size=self.u_net_resolution, num_classes=1).cuda()
        snapshot = os.path.join(self.unet_checkpoint_dir, 'swin_unet_checkpoint.pth')
        msg = self.unet.load_state_dict(torch.load(snapshot))
        print("self trained swin unet",msg)
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
                       obs: Dict,
                       prompts: List[str]):
        self.index += 1
        self.curr_frame = self.get_curr_frame(obs) # shape: [160, 256, 3]
        random_lable = np.random.rand(self.u_net_resolution, self.u_net_resolution, 1)
        img, _ = self.preprocess(self.curr_frame, random_lable)
        img = img.unsqueeze(0) # shape: [1, 3, 224, 224]

        with torch.no_grad():
            texts_feats = self._get_text_feats(prompts).cuda().float()
            img = img.cuda().float().expand(texts_feats.shape[0], -1, -1, -1)
            out = self.unet(img, texts_feats) # out.shape: [P, 1, 224, 224]
            out_np = out.squeeze(1).cpu().detach().numpy() # out_np.shape: [P, 224, 224]
            out_np = out_np.transpose((1, 2, 0))
            out_np = resized_if_need(out_np, target_size=(self.resolution[1], self.resolution[0]))
            out_np = out_np.transpose((2, 0, 1)).squeeze(0)
            out_np = cv2.GaussianBlur(out_np, (self.blur_x, self.blur_y), 0)
            out_np = out_np[np.newaxis, :]

        return out_np

    def _generate_gaussian_distribution(self, height=160, width=256, peak=1.0, sigma_x=128.0, sigma_y=80.0):
        if sigma_x == 0 or sigma_y == 0:
            return np.full((height, width), peak)

        x = np.linspace(-width // 2, width // 2, width)
        y = np.linspace(-height // 2, height // 2, height)
        x, y = np.meshgrid(x, y)
        gaussian = peak * np.exp(-((x**2 / (2 * sigma_x**2)) + (y**2 / (2 * sigma_y**2))))

        return gaussian

    def get_reward(
            self,
            obs: Dict,
            prompts: List[str],
            episode_num: int,
            step_num: int
    ):
        masks = self._generate_mask(obs, prompts)
        self.mask = np.max(masks, axis=0) * 255.0 # shape: [160, 256]

        score = 0
        for mask in masks:
            score += (np.mean(mask * self.gaussian)/self.gaussian_mean)

        self.gaussian_score = score
        self.gaussian_buffer.add(self.gaussian_score)
        heatmap_normalized = self.mask / 255.0

        kurtosis_value = kurtosis(heatmap_normalized.flatten())
        normalized_kurtosis = sigmoid(kurtosis_value)

        self.zoom_in_prob = normalized_kurtosis * (np.max(heatmap_normalized) - np.mean(heatmap_normalized))
        self.check_threshold_buffer.add(self.zoom_in_prob)
        self.check_threshold = self.check_threshold_buffer.get_threshold()   
        
        return score, self.zoom_in_prob, self.check_threshold
        
    def get_heatmap(self, is_zoomed=False):
        if is_zoomed:
            return np.expand_dims(self.mask_on_zoomed_image, axis=-1) # [H, W, 1]
        else:
            return np.expand_dims(self.mask, axis=-1) / 255.0 # [H, W, 1]

    def generate_zoom_in_frame(self, ):
        if self.check_threshold >= self.zoom_in_prob:
            return self.curr_frame, False
        
        image_tensor = torch.from_numpy(self.curr_frame).unsqueeze(0).float().to(self.device) # shape: [1, 160, 256, 3]
        B, H, W, C = image_tensor.shape
        heatmap_normalized = self.mask / 255.0
        threshold_value = (np.max(heatmap_normalized) + np.min(heatmap_normalized)) / 2.0 + np.std(heatmap_normalized)
        _, binary_image = cv2.threshold(heatmap_normalized, threshold_value, 1, cv2.THRESH_BINARY)
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (16, 10))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, open_kernel)
        contours, _ = cv2.findContours(binary_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return self.curr_frame, False

        max_mean_value = 0
        max_area_ratio = 0
        centroid_x = 0
        centroid_y = 0
        self.have_center = False

        for contour in contours:
            mask = np.zeros_like(heatmap_normalized)
            cv2.drawContours(mask, [contour], -1, (1), thickness=cv2.FILLED)
            mean_val = np.mean(heatmap_normalized[mask == 1])
            M = cv2.moments(contour)
            if self.gaussian[int(M['m01'] / M['m00']), int(M['m10'] / M['m00'])] >= self.gaussian_mean:
                self.have_center = True
            if mean_val > max_mean_value:
                max_mean_value = mean_val
                max_area_ratio = np.sum(mask) / mask.size
                centroid_x = int(M['m10'] / M['m00'])
                centroid_y = int(M['m01'] / M['m00'])

        self.num_above_threshold = max_area_ratio / (self.gaussian[centroid_y, centroid_x])
        proportion = torch.tensor([self.num_above_threshold], device=self.device)
        proportion = torch.clamp(proportion, 0.01, 1.0)
        sqrt_proportion = torch.sqrt(proportion) # shape: [1]
        mask = torch.from_numpy(self.mask / 255.0).unsqueeze(0).float().to(self.device) # shape: [1, 160, 256]

        window_width = (sqrt_proportion * W).int()
        window_height = (sqrt_proportion * H).int()
        cur_window_height, cur_window_width = window_height[0], window_width[0]
        kernel = torch.ones((1, 1, cur_window_height, cur_window_width), device=self.device) 
        kernel_size = cur_window_height * cur_window_width
        
        mask = mask.unsqueeze(1) # shape: [1, 1, 160, 256]
        conv_result = F.conv2d(mask, kernel, stride=self.stride)

        best_value, best_idx = torch.max(conv_result.view(-1), 0)
        self.best_value_on_mask = (best_value / kernel_size).item()
        best_y, best_x = divmod(best_idx.item(), conv_result.shape[-1])

        left_top = torch.tensor([best_x, best_y], device=self.device) * self.stride
        right_bottom = left_top + torch.tensor([cur_window_width, cur_window_height], device=self.device)
        window = image_tensor[:, int(left_top[1]):int(right_bottom[1]), int(left_top[0]):int(right_bottom[0]), :] 
        window = window.permute(0, 3, 1, 2) # shape: [1, 3, window_height, window_width]
        zoomed_image = F.interpolate(window, size=(H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1) # shape: [1, 160, 256, 3]
        self.zoomed_frame = zoomed_image.squeeze(0).detach().cpu().numpy().astype(np.uint8) # shape: [160, 256, 3]

        return self.zoomed_frame, True
    
    def compute_reward_on_zoomed_image(self):
        random_lable = np.random.rand(self.u_net_resolution, self.u_net_resolution, 1)
        img, _ = self.preprocess(self.zoomed_frame, random_lable)
        img = img.unsqueeze(0)

        with torch.no_grad():
            texts_feats = self._get_text_feats(self.prompts).cuda().float()
            img = img.cuda().float().expand(texts_feats.shape[0], -1, -1, -1)

            out = self.unet(img, texts_feats) # out.shape: [P, 1, 224, 224]
            out_np = out.squeeze(1).cpu().detach().numpy() # out_np.shape: [P, 224, 224]
            out_np = out_np.transpose((1, 2, 0))
            out_np = resized_if_need(out_np, target_size=(self.resolution[1], self.resolution[0]))
            out_np = out_np.transpose((2, 0, 1)).squeeze(0)
            out_np = cv2.GaussianBlur(out_np, (self.blur_x, self.blur_y), 0)
            out_np = out_np[np.newaxis, :]

        self.mask_on_zoomed_image = np.max(out_np, axis=0)

        zoomed_gaussian = 0
        for mask in out_np:
            zoomed_gaussian += (np.mean(mask * self.gaussian)/self.gaussian_mean)

        kurtosis_value = kurtosis(self.mask_on_zoomed_image.flatten())
        normalized_kurtosis = sigmoid(kurtosis_value)
        zoom_in_prob_on_zoomed_image = normalized_kurtosis * (np.max(self.mask_on_zoomed_image) - np.mean(self.mask_on_zoomed_image))
        zoomed_reward = self.best_value_on_mask

        if zoomed_gaussian < self.gaussian_score + 2.0 * self.gaussian_buffer.std_dev():
            is_zoomed = False
        else:
            is_zoomed = True

        jump = is_zoomed and self.have_center
        
        return zoomed_reward, zoomed_gaussian, zoom_in_prob_on_zoomed_image, is_zoomed, jump
    
    def save_img_and_mask(self):
        print(self.index)
        save_image(self.curr_frame, self.index)
        save_mask(np.expand_dims(self.mask / 255.0, axis=0), self.index)
        save_image(self.zoomed_frame, self.index, name='zoomed')
        save_mask(np.expand_dims(self.mask_on_zoomed_image, axis=0), self.index, name='zoomed_mask')


class MCUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(MCUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = MultimodalSwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                text_feature_dim=config.MODEL.TEXT_FEATURE_DIM,
                                heads=config.MODEL.HEADS)

    def forward(self, x, p):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        logits = self.swin_unet(x, p)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x= self.norm(x)

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        upsample (nn.Module | None, optional): upsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


import torch.nn.functional as F

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
    

class TextImageAttention(nn.Module):
    def __init__(self, image_feature_dim, text_feature_dim, n_heads):
        super().__init__()
        # Assuming that the text_feature_dim can be divided by n_heads
        self.text_projection = nn.Linear(text_feature_dim, image_feature_dim)
        self.multi_head_attention = MultiHeadAttention(n_heads, image_feature_dim, image_feature_dim, image_feature_dim)

    def forward(self, image_features, text_features):
        residual = image_features
        text_features = self.text_projection(text_features) # shape: (batch_size, image_feature_dim)
        text_features = text_features.unsqueeze(1).expand(-1, image_features.shape[1], -1) # shape: (batch_size, height * width, image_feature_dim)
        attention, _ = self.multi_head_attention(q=text_features, k=image_features, v=image_features) # shape: (batch_size, height * width, image_feature_dim)
        return attention

class MultimodalSwinTransformerSys(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None, text_feature_dim=512,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, heads=8,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        assert num_classes == 1, "num_classes should be 1 for heatmap generation!"

        print("SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(depths,
        depths_decoder,drop_path_rate,num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # build fusion layers
        self.TIA = nn.ModuleList()
        for i_layer in range(self.num_layers):
            fusion_layer = TextImageAttention(image_feature_dim=int(embed_dim*2**i_layer), 
                                     text_feature_dim=text_feature_dim, 
                                     n_heads=heads)
            self.TIA.append(fusion_layer)
            
        
        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.num_classes,kernel_size=1,bias=False)
            self.output_sigmoid = nn.Sigmoid()


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
  
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_fusion):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_fusion[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":
            x = self.up(x)
            x = x.view(B,4*H,4*W,-1)
            x = x.permute(0,3,1,2) #B,C,H,W
            x = self.output(x)
            x = self.output_sigmoid(x)
            
        return x

    def fusion(self, x_downsample, p):
        x_fusion = []
        for inx, fusion_layer in enumerate(self.TIA):
            if inx > 1:
                attention = fusion_layer(x_downsample[inx], p)
                x_fusion.append(attention)
            else:
                x_fusion.append(x_downsample[inx])
        
        return x_fusion

    def forward(self, x, p):
        x, x_downsample = self.forward_features(x)
        x_fusion = self.fusion(x_downsample, p)
        x = self.forward_up_features(x, x_fusion)
        x = self.up_x4(x)

        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops