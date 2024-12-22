from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, dense_reward=.01, smoothing=1, target_object='log', **kwargs):
        super().__init__(env)
        self.clip = clip # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.expl_prompt = [f"Explore the widest possible area to find {target_object}"]
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        
        self.buffer = None
        self._clip_state = None, None
        self.expl_buffer = None
        self._expl_clip_state = None, None
        self.last_score = 0
        self.expl_last_score = 0

    def reset(self, **kwargs):
        self._clip_state = None, self._clip_state[1]
        self._expl_clip_state = None, self._expl_clip_state[1]
        self.buffer = None
        self.expl_buffer = None
        self.last_score = 0
        self.expl_last_score = 0

        obs = self.env.reset(**kwargs)
        obs['intrinsic'] = 0.0
        obs['score'] = 0.0

        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            logits, self._clip_state = self.clip.get_logits(obs, self.prompt, self._clip_state)
            logits = logits.detach().cpu()

            self.buffer = self._insert_buffer(self.buffer, logits[:1])
            score = self._get_score()

            if score > self.last_score:
                obs['intrinsic'] = self.dense_reward * score
                self.last_score = score
            else:
                obs['intrinsic'] = 0.0

            obs['score'] = self.dense_reward * score

        else:
            obs['intrinsic'] = 0.0
            obs['score'] = 0.0

        if len(self.expl_prompt) > 0:
            logits, self._expl_clip_state = self.clip.get_logits(obs, self.expl_prompt, self._expl_clip_state)
            logits = logits.detach().cpu()

            self.expl_buffer = self._insert_buffer(self.expl_buffer, logits[:1])
            expl_score = self._get_expl_score()

            if expl_score > self.expl_last_score:
                info['expl_intrinsic'] = self.dense_reward * expl_score
                self.expl_last_score = expl_score
            else:
                info['expl_intrinsic'] = 0.0

        else:
            info['expl_intrinsic'] = 0.0

        info["clip_score"] = obs['intrinsic']
        info["clip_last_score"] = self.last_score
        info["clip_dense_reward"] = self.dense_reward    

        return obs, reward, done, info 

    def _get_score(self):
        score = th.mean(self.buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()
    
    def _get_expl_score(self):
        score = th.mean(self.expl_buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()

    def _insert_buffer(self, buffer, logits):
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
