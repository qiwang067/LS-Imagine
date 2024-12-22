from typing import List, Tuple, Dict
import torch as th
from omegaconf import OmegaConf
from mineclip import MineCLIP
from abc import ABC, abstractstaticmethod


class ClipReward(ABC):
    def __init__(self, ckpt="weights/mineclip_attn.pth", **kwargs) -> None:
        kwargs["arch"] = kwargs.pop("arch", "vit_base_p16_fz.v2.t2")
        kwargs["hidden_dim"] = kwargs.pop("hidden_dim", 512)
        kwargs["image_feature_dim"] = kwargs.pop("image_feature_dim", 512)
        kwargs["mlp_adapter_spec"] = kwargs.pop("mlp_adapter_spec", "v0-2.t0")
        kwargs["pool_type"] = kwargs.pop("pool_type", "attn.d2.nh8.glusw")
        kwargs["resolution"] = [160, 256]

        self.resolution = self.get_resolution()
        self.device = kwargs.pop("device", "cuda")
        self.model = None
        
        self._load_mineclip(ckpt, kwargs)

    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()

    def _load_mineclip(self, ckpt, config):
        config = OmegaConf.create(config)
        self.model = MineCLIP(**config).to(self.device)
        self.model.load_ckpt(ckpt, strict=True)
        if self.resolution != (160, 256):  # Not ideal, but we need to resize the relative position embedding
            self.model.clip_model.vision_model._resolution = th.tensor([160, 256])  # This isn't updated from when mineclip resized it
            self.model.clip_model.vision_model.resize_pos_embed(self.resolution)
        self.model.eval()

    def _get_reward_from_logits(
        self,
        logits: th.Tensor  # P
    ) -> float:
        probs = th.softmax(logits, 0)
        return max(probs[0].item() - 1 / logits.shape[0], 0)

    def _get_image_feats(
        self,
        curr_frame: th.Tensor,
        past_frames: th.Tensor = None
    ) -> th.Tensor:
        while len(curr_frame.shape) < 5:
            curr_frame = curr_frame.unsqueeze(0)
        assert curr_frame.shape == (1, 1, 3) + self.resolution, "Found shape {}".format(curr_frame.shape)
        curr_frame_feats = self.model.forward_image_features(curr_frame.to(self.device))  # 1 x 1 x 512

        if past_frames is None:
            past_frames = th.zeros((15, curr_frame_feats.shape[-1]))
        past_frames = past_frames.to(self.device)

        while len(past_frames.shape) < 3:
            past_frames = past_frames.unsqueeze(0)
        assert past_frames.shape == (1, 15, curr_frame_feats.shape[-1]), "Found shape {}".format(past_frames.shape)

        return th.cat((past_frames, curr_frame_feats), dim=1)
        
    def _get_video_feats(
        self,
        image_feats: th.Tensor
    ) -> th.Tensor:
        return self.model.forward_video_features(image_feats.to(self.device))  # 1 x 512

    def _get_text_feats(
        self,
        prompts: str
    ) -> th.Tensor:
        text_feats = self.model.encode_text(prompts)  # P x 512
        assert len(text_feats.shape) == 2 and text_feats.shape[0] == len(prompts), "Found shape {}".format(text_feats.shape)
        return text_feats

    def get_logits(
        self,
        obs: Dict,  # 3 x 160 x 256
        prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        curr_frame = self._get_curr_frame(obs)
        past_frames, text_feats = state

        with th.no_grad():
            if text_feats is None:
                text_feats = self._get_text_feats(prompts)

            image_feats = self._get_image_feats(curr_frame, past_frames)
            video_feats = self._get_video_feats(image_feats)
            logits = self.model.forward_reward_head(video_feats.to(self.device), text_tokens=text_feats.to(self.device))[0][0]  # P

        return logits, (image_feats[0, 1:].cpu(), text_feats.cpu())

    def get_reward(
        self, 
        obs: Dict,
        prompt: str,
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[float, Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            [prompt] + neg_prompts,
            state
        )
        reward = self._get_reward_from_logits(logits)

        return reward, state

    def get_rewards(
        self, 
        obs: Dict,
        prompts: List[str],
        neg_prompts: List[str],
        state: Tuple[th.Tensor, th.Tensor] = None  # history x 512
    ) -> Tuple[List[float], Tuple[th.Tensor, th.Tensor]]:
        logits, state = self.get_logits(
            obs,
            prompts + neg_prompts,
            state
        )
        rewards = []
        for i in range(len(prompts)):
            rewards.append(self._get_reward_from_logits(th.cat((
                logits[i:i+1],
                logits[len(prompts):]
            ))))
        return rewards, state
