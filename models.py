import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from torch import nn

import networks
import tools


to_np = lambda x: x.detach().cpu().numpy()

def probability_to_bool(input_tensor, jump_prob=1.0):
    random_tensor = torch.rand_like(input_tensor)
    random_mask = random_tensor < jump_prob # bool
    
    bool_tensor = input_tensor > 0.5 # bool
    bool_tensor = bool_tensor & random_mask
    
    return bool_tensor

class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()

class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()} # 'image': (64, 64, 3)
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )

        self.heads = nn.ModuleDict()

        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )

        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )

        self.heads["end"] = networks.MLP(
            feat_size,
            (),
            config.end_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.end_head["outscale"],
            device=config.device,
            name="End",
        )

        self.heads["jump"] = networks.MLP(
            feat_size,
            (),
            config.jump_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.jump_head["outscale"],
            device=config.device,
            name="Jump",
        )

        self.heads["intrinsic"] = networks.MLP(
            feat_size,
            (255,) if config.intrinsic_head["dist"] == "symlog_disc" else (),
            config.intrinsic_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.intrinsic_head["dist"],
            outscale=config.intrinsic_head["outscale"],
            device=config.device,
            name="Intrinsic",
        )

        self.heads["jumping_steps"] = networks.MLP(
            feat_size * 2,
            (255,) if config.jumping_steps_head["dist"] == "symlog_disc" else (),
            config.jumping_steps_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.jumping_steps_head["dist"],
            outscale=config.jumping_steps_head["outscale"],
            device=config.device,
            name="jumping_steps",
        )

        self.heads["accumulated_reward"] = networks.MLP(
            feat_size * 2,
            (255,) if config.accumulated_reward_head["dist"] == "symlog_disc" else (),
            config.accumulated_reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.accumulated_reward_head["dist"],
            outscale=config.accumulated_reward_head["outscale"],
            device=config.device,
            name="accumulated_reward",
        )
       
        for name in config.grad_heads:
            assert name in self.heads, name

        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )

        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )

        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            end=config.end_head["loss_scale"],
            jump=config.jump_head["loss_scale"],
            intrinsic=config.intrinsic_head["loss_scale"],
            jumping_steps=config.jumping_steps_head["loss_scale"],
            accumulated_reward=config.accumulated_reward_head["loss_scale"],
        )

    def _train(self, data_origin):
        
        data = self.preprocess(data_origin, zoomed=False)
        data_zoomed = self.preprocess(data_origin, zoomed=True)

        zoomed_num = torch.sum(data["is_zoomed"]).item()
        calculated_num = torch.sum(data_zoomed["is_calculated"]).item()

        with tools.RequiresGrad(self):
            with torch.cuda.amp.autocast(self._use_amp):
                
                embed = self.encoder(data)
                embed_zoomed = self.encoder(data_zoomed)

                # process original data
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                kl_free = self._config.kl_free # 1.0
                dyn_scale = self._config.dyn_scale # 0.5
                rep_scale = self._config.rep_scale # 0.1

                kl_loss_img, kl_value_img, dyn_loss_img, rep_loss_img = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )

                assert kl_loss_img.shape == embed.shape[:2], kl_loss_img.shape

                preds = {}
                for name, head in self.heads.items():
                    # When processing original data, "jumping_steps" and "accumulated_reward" are not used
                    if name == "jumping_steps" or name == "accumulated_reward":
                        continue
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                        
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                    
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }

                if zoomed_num > 0:
                    # process zoomed data
                    is_zoomed_indices = data["is_zoomed"].squeeze(-1).bool()
                    is_calculated_mask = data["is_calculated"][is_zoomed_indices]
                    embed_zoomed = embed_zoomed[is_zoomed_indices].unsqueeze(1)
                    for key, value in data_zoomed.items():
                        data_zoomed[key] = value[is_zoomed_indices].unsqueeze(1)

                    selected_post = dict()
                    selected_prior = dict()
                    
                    for key, value in post.items():
                        selected_post[key] = value[is_zoomed_indices].unsqueeze(1)

                    for key, value in prior.items():
                        selected_prior[key] = value[is_zoomed_indices].unsqueeze(1)

                    post_zoomed, prior_zoomed = self.dynamics.observe_zoomed(
                        embed_zoomed, data_zoomed["action"], data_zoomed["is_first"], selected_post, selected_prior
                    )
                    
                    kl_loss_jmp, kl_value_jmp, dyn_loss_jmp, rep_loss_jmp = self.dynamics.kl_loss(
                        post_zoomed, prior_zoomed, kl_free, dyn_scale, rep_scale
                    )

                    assert kl_loss_jmp.shape == embed_zoomed.shape[:2], kl_loss_jmp.shape

                    preds_zoomed = {}
                    for name, head in self.heads.items():
                        grad_head_zoomed = name in self._config.grad_heads

                        if name == "jumping_steps" or name == "accumulated_reward":
                            feat_zoomed = self.dynamics.get_feat(post_zoomed)
                            feat_before_zoom = self.dynamics.get_feat(selected_post)
                            feat_concat = torch.concat([feat_before_zoom, feat_zoomed], dim=-1)

                            feat_concat = feat_concat if grad_head_zoomed else feat_concat.detach()
                            pred_zoomed = head(feat_concat)

                        else:
                            feat_zoomed = self.dynamics.get_feat(post_zoomed)
                            feat_zoomed = feat_zoomed if grad_head_zoomed else feat_zoomed.detach()
                            pred_zoomed = head(feat_zoomed)
                            
                        if type(pred_zoomed) is dict:
                            preds_zoomed.update(pred_zoomed)
                        else:
                            preds_zoomed[name] = pred_zoomed
                              
                    losses_zoomed = {}
                    for name, pred in preds_zoomed.items():
                        if name == 'jumping_steps' or name == 'accumulated_reward':
                            loss = -pred.log_prob(data_zoomed[name])
                            loss *= is_calculated_mask
                            if loss.shape[1] != 1:
                                loss = loss.mean(dim=1, keepdim=True)
                            assert loss.shape == embed_zoomed.shape[:2], (name, loss.shape)
                            losses_zoomed[name] = loss
                            
                        else:
                            loss = -pred.log_prob(data_zoomed[name])
                            if loss.shape[1] != 1:
                                loss = loss.mean(dim=1, keepdim=True)
                            assert loss.shape == embed_zoomed.shape[:2], (name, loss.shape)
                            losses_zoomed[name] = loss
                        
                    scaled_zoomed = {
                        key: value * self._scales.get(key, 1.0)
                        for key, value in losses_zoomed.items()
                    }

                if zoomed_num > 0:
                    kl_loss_img = kl_loss_img.reshape(-1, *kl_loss_img.shape[2:])
                    kl_loss_jmp = kl_loss_jmp.reshape(-1, *kl_loss_jmp.shape[2:])
                    kl_loss = torch.cat((kl_loss_img, kl_loss_jmp), dim=0)

                    kl_value_img = kl_value_img.reshape(-1, *kl_value_img.shape[2:])
                    kl_value_jmp = kl_value_jmp.reshape(-1, *kl_value_jmp.shape[2:])
                    kl_value = torch.cat((kl_value_img, kl_value_jmp), dim=0)

                    dyn_loss_img = dyn_loss_img.reshape(-1, *dyn_loss_img.shape[2:])
                    dyn_loss_jmp = dyn_loss_jmp.reshape(-1, *dyn_loss_jmp.shape[2:])
                    dyn_loss = torch.cat((dyn_loss_img, dyn_loss_jmp), dim=0)

                    rep_loss_img = rep_loss_img.reshape(-1, *rep_loss_img.shape[2:])
                    rep_loss_jmp = rep_loss_jmp.reshape(-1, *rep_loss_jmp.shape[2:])
                    rep_loss = torch.cat((rep_loss_img, rep_loss_jmp), dim=0)

                    scaled_img = sum(scaled.values()).reshape(-1, *sum(scaled.values()).shape[2:]) # [512]
                    scaled_jmp = sum(scaled_zoomed.values()).reshape(-1, *sum(scaled_zoomed.values()).shape[2:]) # [N]
                    scaled_sum = torch.cat((scaled_img, scaled_jmp * self._config.long_term_branch_weight), dim=0) # [512 + N]
                    
                    model_loss = scaled_sum + kl_loss

                else:
                    kl_loss = kl_loss_img
                    kl_value = kl_value_img
                    dyn_loss = dyn_loss_img
                    rep_loss = rep_loss_img
                    
                    model_loss = sum(scaled.values()) + kl_loss

            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(torch.mean(loss)) for name, loss in losses.items()})
        if zoomed_num > 0:
            metrics.update({f"zoomed_{name}_loss": to_np(torch.mean(loss)) for name, loss in losses_zoomed.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(torch.mean(dyn_loss))
        metrics["rep_loss"] = to_np(torch.mean(rep_loss))
        metrics["kl"] = to_np(torch.mean(kl_value))
        if zoomed_num > 0:
            metrics["dyn_loss_img"] = to_np(torch.mean(dyn_loss_img))
            metrics["dyn_loss_jmp"] = to_np(torch.mean(dyn_loss_jmp))
            metrics["rep_loss_img"] = to_np(torch.mean(rep_loss_img))
            metrics["rep_loss_jmp"] = to_np(torch.mean(rep_loss_jmp))
        metrics["model_loss"] = to_np(torch.mean(model_loss))

        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            if zoomed_num > 0:
                metrics["prior_zoomed_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(prior_zoomed).entropy())
                )
                metrics["post_zoomed_ent"] = to_np(
                    torch.mean(self.dynamics.get_dist(post_zoomed).entropy())
                )    
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )

        post = {k: v.detach() for k, v in post.items()}

        if zoomed_num > 0:
            post_zoomed = {k: v.detach() for k, v in post_zoomed.items()}
            return post, post_zoomed, context, metrics
        else:
            return post, None, context, metrics

    # this function is called during both rollout and training
    def preprocess(self, obs, zoomed=False):
        obs = obs.copy()

        if not zoomed:
            obs["image"] = torch.Tensor(obs["image"]) / 255.0
            obs["heatmap"] = torch.Tensor(obs["heatmap"]).unsqueeze(-1) / 255.0
            if "action" in obs:
                original_action = obs["action"]
                zeros_array = np.zeros((original_action.shape[0], original_action.shape[1], 1), dtype=original_action.dtype)
                new_action = np.concatenate((original_action, zeros_array), axis=-1)
                obs["action"] = new_action  # [16, 64, 13]

        else:
            obs["is_zoomed"] = np.zeros_like(obs["is_zoomed"])
            obs["jump"] = np.zeros_like(obs["jump"])
            obs["image"] = torch.Tensor(obs["zoomed_image"]) / 255.0
            obs["heatmap"] = torch.Tensor(obs["heatmap_on_zoomed"]).unsqueeze(-1) / 255.0
            obs["reward"] = obs["reward_on_zoomed"]
            obs['intrinsic'] = obs['intrinsic_on_zoomed']
            
            # for states after zooming, clear the action and add it to 13th dimension, and set the last dimension to 1
            if "action" in obs:
                new_action = np.zeros((obs["action"].shape[0], obs["action"].shape[1], obs["action"].shape[2] + 1), dtype=obs["action"].dtype) # [64, 16, 13]
                new_action[:, :, -1] = 1
                obs["action"] = new_action
            
        obs.pop("zoomed_image", None)
        obs.pop("heatmap_on_zoomed", None)
        obs.pop("reward_on_zoomed", None)
        obs.pop("intrinsic_on_zoomed", None)

        if "discount" in obs:
            obs["discount"] *= self._config.discount
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train end_head
        assert "is_terminal" in obs

        if zoomed:
            obs['is_first'] = np.zeros_like(obs['is_first'])

        obs["is_zoomed"] = torch.Tensor(obs["is_zoomed"]).unsqueeze(-1)
        obs["jump"] = torch.Tensor(obs["jump"]).unsqueeze(-1)
        obs["is_calculated"] = torch.Tensor(obs["is_calculated"]).unsqueeze(-1)
        obs["end"] = torch.Tensor(obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data, zoomed=False)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
    
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]

        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self.jump_prob = config.jump_prob
        self.gamma_sum = [(1 - self._config.discount ** (i + 1)) / (1 - self._config.discount) for i in range(self._config.episode_max_steps)]
        self.gamma_sum = torch.tensor(self.gamma_sum, dtype=torch.float32, device=config.device)

        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        start_zoomed,
        objective,
        intrinsic_objective,
        jumping_steps_predictor,
        accumulated_reward_predictor,
        jump_indicator,
        is_end,
    ):

        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                # add post-jump state to start
                flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
                start = {k: flatten(v) for k, v in start.items()} # [512, xx, xx]
                if start_zoomed is not None and self.jump_prob > 0.0:
                    start_zoomed = {k: flatten(v) for k, v in start_zoomed.items()} # [n, xx, xx]
                    for k, v in start.items():
                        start[k] = torch.cat((v, start_zoomed[k]), dim=0) # shape: [N = 512 + n, xx, xx]

                state_num = start['deter'].shape[0] # 512 or N

                imag_state = {} # [L, N, xx, xx]
                for k, v in start.items():
                    imag_state[k] = v.unsqueeze(0) # [1, N, xx, xx]

                action_example = self.actor(self._world_model.dynamics.get_feat(start).detach()).sample()
                action_dimension = action_example.shape[-1]

                jump_record = torch.empty((0, state_num), device=self._config.device) # [0, N]
                imag_action = torch.empty((0, state_num, action_dimension), device=self._config.device) # [0, N, xx]

                for _ in range (self._config.imag_horizon - 1):
                    checking_state = {} # [N, xx, xx]
                    for key, tensor in imag_state.items():
                        checking_state[key] = tensor[-1, :, ...]

                    # check if checking_state can jump
                    jump_tensor = jump_indicator(checking_state) # [N, 1]
                    end_factor = is_end(checking_state) # [N, 1]
                    indices = probability_to_bool(jump_tensor * (1.0 - end_factor), self.jump_prob).squeeze() # [N]
                    jump_record = torch.cat((jump_record, indices.unsqueeze(0)), dim=0) # [L, N]

                    jump_state = {} # [X, xx, xx]
                    for key, tensor in checking_state.items():
                        jump_state[key] = tensor[indices]

                    # for states identified as requiring a jumpy transition, execute long-term imagination.
                    _, state_after_jumping, _ = self._jumpy(
                        jump_state, self.actor, 1
                    )

                    _, state_after_jumping, _ = self._imagine(
                        state_after_jumping, self.actor, 1
                    ) # [X, xx, xx]

                    # For other states, execute short-term imagination.
                    _, state_after_imagination, ac = self._imagine(
                        checking_state, self.actor, 1
                    ) # [N, xx, xx]

                    # save action for this step to imag_action
                    imag_action = torch.cat((imag_action, ac.unsqueeze(0)), dim=0)
 
                    for key in state_after_imagination:
                        state_after_imagination[key][indices] = state_after_jumping[key]

                    for key in imag_state:
                        imag_state[key] = torch.cat((imag_state[key], state_after_imagination[key].unsqueeze(0)), dim=0)


                last_jump_record = torch.zeros((1, state_num), device=self._config.device) # [1, N]
                jump_record = torch.cat((jump_record, last_jump_record), dim=0) # [L, N]
                jump_num = torch.sum(jump_record)

                imag_feat = self._world_model.dynamics.get_feat(imag_state) # [L, N, xx, xx]
                inp = imag_feat[-1].detach()
                last_imag_action = self.actor(inp).sample() # [N, xx, xx]
                imag_action = torch.cat((imag_action, last_imag_action.unsqueeze(0)), dim=0) # [L, N, xx]

                #  Data augmentation (using the state after long-term transition as the starting point for a new imagination sequence)
                new_state = {}

                new_jump_tensor = jump_indicator(imag_state) # [L, N, 1]
                new_end_factor = is_end(imag_state) # [L, N, 1]
                zoom_indices = new_jump_tensor * (1.0 - new_end_factor)

                max_values, _ = zoom_indices.max(dim=0, keepdim=True)
                max_mask = (zoom_indices == max_values)
                zoom_indices *= max_mask.float()
                zoom_indices = probability_to_bool(zoom_indices, self.jump_prob).squeeze() # [L, N]

                new_num = torch.sum(zoom_indices) # Y

                for key, tensor in imag_state.items():
                    new_state[key] = tensor[zoom_indices] # [Y, xx, xx]

                _, new_state_after_jump, _ = self._jumpy(
                    new_state, self.actor, 1
                ) 

                _, new_state_after_jump, _ = self._imagine(
                    new_state_after_jump, self.actor, 1
                ) 

                new_feat, new_state_sequence, new_action = self._imagine(
                    new_state_after_jump, self.actor, self._config.imag_horizon
                ) # [L, N, xx, xx]

                new_jump_record = torch.zeros((self._config.imag_horizon, new_num), device=self._config.device) # [L, Y]

                for key, tensor in imag_state.items():
                    imag_state[key] = torch.cat((tensor, new_state_sequence[key]), dim=1) # [L, N+Y, xx, xx]

                imag_feat = torch.cat((imag_feat, new_feat), dim=1) # [L, N+Y, xx]
                imag_action = torch.cat((imag_action, new_action), dim=1) # [L, N+Y, 12]
                jump_record = torch.cat((jump_record, new_jump_record), dim=1) # [L, N+Y]

                imagination_num_tensor = torch.tensor(state_num, dtype=torch.float32, device=imag_feat.device)
                
                reward = objective(imag_feat, imag_state, imag_action)
                
                intrinsic_reward = intrinsic_objective(imag_feat, imag_state, imag_action)
                reward += intrinsic_reward


                actor_ent = self.actor(imag_feat).entropy() 
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()

                jump_sequence_record = torch.any(jump_record, dim=0, keepdim=True).repeat(jump_record.size(0), 1) # [L, N]
                jump_record_tensor = jump_record.unsqueeze(-1) # [L, N, 1]
                jump_sequence_record = jump_sequence_record.unsqueeze(-1) # [L, N, 1]

                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward, jump_record_tensor, jumping_steps_predictor, accumulated_reward_predictor, is_end
                )

                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                    jump_record_tensor,
                )

                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)

                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        metrics.update(tools.tensorstats(imagination_num_tensor, "imagination_num"))

        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def save_state_sequence(self, imag_feat, jump_record, freq=0.1):
        if random.random() > freq:
            return

        output_dir = os.path.join(self._config.logdir, "check_obs")
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

        num_rows, num_cols, _ = jump_record.shape
        if torch.sum(jump_record) == 0:
            random_num = 0.0
        else:
            random_num = np.random.rand()

        if random_num < 0.5:
            random_index = np.random.randint(num_cols)
        else:
            valid_indices = torch.any(jump_record.squeeze(-1), dim=0).nonzero().squeeze()

            if valid_indices.ndim == 0: 
                random_index = valid_indices.item()
            else:
                random_index = valid_indices[torch.randint(len(valid_indices), (1,)).item()].item()

        feats = imag_feat[:, random_index, :] 
        jump_record = jump_record[:, random_index]  

        def tensor_to_image(tensor):
            array = tensor.detach().cpu().numpy().astype(np.uint8)
            return Image.fromarray(array)

        def apply_colormap_to_heatmap(heatmap_tensor, cmap='jet', vmin=0, vmax=1):
            heatmap = heatmap_tensor.detach().cpu().numpy().squeeze()
            normed_heatmap = (heatmap - vmin) / (vmax - vmin) 
            normed_heatmap = np.clip(normed_heatmap, 0, 1) 
            colormap = plt.get_cmap(cmap)
            colored_heatmap = colormap(normed_heatmap) 
            colored_heatmap = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
            return Image.fromarray(colored_heatmap)

        def blend_images(image, heatmap, alpha=0.5):
            image = image.convert("RGB")
            heatmap = heatmap.convert("RGB")
            return Image.blend(image, heatmap, alpha)

        image_list = []
        for feat in feats:
            feat = feat.unsqueeze(0).unsqueeze(0)

            image_tensor = self._world_model.heads["decoder"](feat)["image"].mode()
            image_tensor = image_tensor.squeeze(0).squeeze(0)
            image_tensor = torch.clamp(image_tensor, min=0.0, max=1.0) * 255.0
            image = tensor_to_image(image_tensor)

            heatmap_tensor = self._world_model.heads["decoder"](feat)["heatmap"].mode()
            heatmap_tensor = heatmap_tensor.squeeze(0).squeeze(0)
            heatmap_tensor = torch.clamp(heatmap_tensor, min=0.0, max=1.0)
            heatmap = apply_colormap_to_heatmap(heatmap_tensor, cmap='jet', vmin=0, vmax=1)

            blended_image = blend_images(image, heatmap, alpha=0.3)

            image_list.append((image, heatmap, blended_image))

        border_width = 2

        total_width = sum(img.width for img, _, _ in image_list) + border_width * len(image_list) * 2
        max_height = max(img.height for img, _, _ in image_list) + border_width * 2
        composite_image = Image.new('RGB', (total_width, max_height * 3), color='white')

        x_offset = 0

        for i, (orig_img, heatmap, blended_img) in enumerate(image_list):
            for row_idx, img in enumerate([orig_img, heatmap, blended_img]):
                y_offset = max_height * row_idx

                if jump_record[i] == 1:
                    border_color = 'red'
                else:
                    border_color = 'blue'

                bordered_img = Image.new('RGB', (img.width + border_width * 2, img.height + border_width * 2), border_color)
                bordered_img.paste(img, (border_width, border_width))

                composite_image.paste(bordered_img, (x_offset, y_offset))

            x_offset += orig_img.width + border_width * 2

        composite_image.save(os.path.join(output_dir, f"{current_time}.png"))
        

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            
            # When interacting with the wm, the action needs to be expanded to 13 dimensions.
            zeros_tensor = torch.zeros(action.shape[0], 1).to(action.device)
            new_action = torch.cat((action, zeros_tensor), dim=-1)
            
            succ = dynamics.img_step(state, new_action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )

        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        if horizon == 1:
            for k, v in succ.items():
                succ[k] = v.squeeze(0)
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)
            return feats, succ, actions
        else:
            return feats, states, actions

    def _jumpy(self, start, policy, horizon):
        dynamics = self._world_model.dynamics

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()

            new_action = torch.zeros(action.shape[0], action.shape[1] + 1).to(action.device)
            new_action[:, -1] = 1
            succ = dynamics.img_step(state, new_action)
            return succ, feat, action
        
        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        if horizon == 1:
            for k, v in succ.items():
                succ[k] = v.squeeze(0)
            feats = feats.squeeze(0)
            actions = actions.squeeze(0)
            return feats, succ, actions
        else:
            return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward, jump_record, jumping_steps_predictor, accumulated_reward_predictor, is_end):
        fc = torch.cat((imag_feat[:-1], imag_feat[1:]), dim=-1) # [L - 1, N, 2xx]
        jumping_steps = jumping_steps_predictor(fc, None, None) # [L - 1, N, 1]
        jumping_steps = torch.cat([jumping_steps, torch.zeros_like(jumping_steps[0]).unsqueeze(0)], dim=0) # [L, N, 1]
        accumulated_reward = accumulated_reward_predictor(fc, None, None) # [L - 1, N, 1]
        accumulated_reward = torch.cat([accumulated_reward, torch.zeros_like(accumulated_reward[0]).unsqueeze(0)], dim=0) # [L, N, 1]
        accumulated_reward *= self.gamma_sum[(jumping_steps - 2).clamp(0, self._config.episode_max_steps - 1)]
        
        end = is_end(imag_state) # [L, N, 1]
        gamma = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        
        jumping_steps = (jumping_steps - 1) * jump_record + 1
        accumulated_reward *= jump_record
        discount = gamma * (1.0 - end)
        
        target = tools.lambda_return_for_ls_imagine(
            reward[1:],
            value[:-1],
            gamma[:-1],
            end[:-1],
            jumping_steps[:-1],
            accumulated_reward[:-1],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        
        self.save_state_sequence(imag_feat, jump_record)
        
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()

        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
        jump_record,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        
        jump_mask = 1.0 - jump_record
        actor_loss = -weights[:-1] * jump_mask[:-1] * actor_target

        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1