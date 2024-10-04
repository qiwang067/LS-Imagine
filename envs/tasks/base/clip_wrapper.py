from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, mineclip=False, dense_reward=.01, clip_target=23, clip_min=21, smoothing=1, **kwargs):
        super().__init__(env)
        self.clip = clip # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.mineclip = mineclip
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        self.clip_target = th.tensor(clip_target)
        self.clip_min = th.tensor(clip_min)
        
        self.buffer = None
        self._clip_state = None, None
        self.last_score = 0

    def reset(self, **kwargs):
        print("==========Now is resetting from ClipWrapper!==========")
        
        self._clip_state = None, self._clip_state[1]
        self.buffer = None
        self.last_score = 0
        print("self.prompt", self.prompt)
        # print("self._clip_state", self._clip_state)
        tmp = self.env.reset(**kwargs)
        tmp['intrinsic'] = 0.0
        tmp['score'] = 0.0
        print("==========Resetting from ClipWrapper is done!==========")
        return tmp
    
    def step(self, action):
        # print("function step() from clip_wrapper.py")
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0 and self.mineclip:
            # print(self.prompt)
            logits, self._clip_state = self.clip.get_logits(obs, self.prompt, self._clip_state)
            logits = logits.detach().cpu()

            self.buffer = self._insert_buffer(self.buffer, logits[:1])
            score = self._get_score()

            # reward += self.dense_reward * score

            # if score > self.last_score:
            #     reward += self.dense_reward * score
            #     self.last_score = score

            if score > self.last_score:
            # if True:
                obs['intrinsic'] = self.dense_reward * score
                self.last_score = score

            else:
                obs['intrinsic'] = 0.0

            obs['score'] = self.dense_reward * score

            # info["clip_score"] = score
            # info["clip_last_score"] = self.last_score
            # info["clip_dense_reward"] = self.dense_reward

        else:
            obs['intrinsic'] = 0.0
            obs['score'] = 0.0

        info["clip_score"] = obs['intrinsic']
        info["clip_last_score"] = self.last_score
        info["clip_dense_reward"] = self.dense_reward

        return obs, reward, done, info 

    def _get_score(self):
        """"""
        # return (max(
        #     th.mean(self.buffer) - self.clip_min,
        #     0
        # ) / (self.clip_target - self.clip_min)).item()

        score = th.mean(self.buffer)
        # print ("score: ", score, (1 / (1 + th.exp(1.2 * (21.8 - score)))).item())
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()

    def _insert_buffer(self, buffer, logits):
        """"""
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
