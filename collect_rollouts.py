import argparse
import functools
import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

from datetime import datetime

from expr import Dreamer
from expr import make_env



def collect_rollouts(config, envs, agent):
    if config.is_random:
        save_dir = config.rollout_from_random_agent_dir
    else:
        save_dir = config.rollout_from_trained_agent_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    step, episode = 0, 0
    done = np.ones(len(envs), bool)
    length = np.zeros(len(envs), np.int32)
    obs = [None] * len(envs)
    agent_state = None
    reward = [0] * len(envs)

    images = os.listdir(save_dir)

    image_num = 0

    while image_num < config.rollout_image_num:
        if done.any():
            indices = [index for index, d in enumerate(done) if d] # 
            results = [envs[i].reset() for i in indices] # 
            results = [r() for r in results] 

            for index, result in zip(indices, results):
                # replace obs with done by initial state
                obs[index] = result

        obs = {k: np.stack([o[k] for o in obs]) for k in obs[0] if "log_" not in k}
        action, agent_state = agent(obs, done, agent_state)

        if isinstance(action, dict):
            action = [
                {k: np.array(action[k][i].detach().cpu()) for k in action}
                for i in range(len(envs))
            ]
        else:
            action = np.array(action)
        assert len(action) == len(envs)

        results = [e.step(a) for e, a in zip(envs, action)]
        results = [r() for r in results]
        obs, reward, done, info = zip(*[p[:4] for p in results])

        obs = list(obs)
        reward = list(reward)
        done = np.stack(done)
        info = list(info)

        images = os.listdir(save_dir)
        image_num = len(images)
        print(image_num)



    

    pass

def generate_mask():
    pass

def finetune_unet():
    pass

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run: # False
        tools.enable_deterministic_run()

    import datetime

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = pathlib.Path(config.logdir).expanduser() 
    rollout_dir = logdir / "rollouts" / config.task
    rollout_dir.mkdir(parents=True, exist_ok=True)
    rollout_from_random_agent_dir = rollout_dir / "image_from_random_agent" / timestamp
    rollout_from_trained_agent_dir = rollout_dir / "image_from_trained_agent" / timestamp
    config.rollout_from_random_agent_dir = rollout_from_random_agent_dir
    config.rollout_from_trained_agent_dir = rollout_from_trained_agent_dir
    rollout_from_random_agent_dir.mkdir(parents=True, exist_ok=True)
    rollout_from_trained_agent_dir.mkdir(parents=True, exist_ok=True)

    make = lambda mode, id: make_env(config, mode, id)

    suite, task = config.task.split("_", 1)
    
    if suite == 'minedojo':
        from envs.tasks import get_specs

        kwargs=dict(
                # log_dir=log_dir,
                target_item=config.target_item
            )
        task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict
        
        config.pmt = task_specs['clip_specs']['prompts']
        config.mineclip_reward = task_specs['clip_specs']['mineclip']
        config.fusion = task_specs['concentration_specs']['fusion']
        task_specs['DV3_specs']['fusion'] = config.fusion
        task_specs['concentration_specs']['mineclip'] = config.mineclip_reward
        task_specs['concentration_specs']['max_steps'] = task_specs['terminal_specs']['max_steps']
        config.episode_max_steps = task_specs['terminal_specs']['max_steps']
        config.gaussian_reward = task_specs['concentration_specs']['gaussian']
        config.zoom_in = task_specs['concentration_specs']['zoom_in']
        
        print("==============================================")
        print("config.pmt", config.pmt)
        print("config.gaussian_reward", config.gaussian_reward)
        print("config.zoom_in", config.zoom_in)
                
        if config.is_random:
            task_specs['screenshot_specs']['save_dir'] = rollout_from_random_agent_dir
        else:
            task_specs['screenshot_specs']['save_dir'] = rollout_from_trained_agent_dir

        print("save_dir", task_specs['screenshot_specs']['save_dir'])
        print("===============================================")

    envs = [make("eval", i) for i in range(config.envs)]

    if config.parallel:
        envs = [Parallel(env, "process") for env in envs]
    else:
        envs = [Damy(env) for env in envs]

    acts = envs[0].action_space
    print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]


    if config.is_random:
        if hasattr(acts, "discrete"):
            print("=====01=====")
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            print("=====02=====")
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )
        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        collect_rollouts(config, envs, random_agent)

    else:
        step = 0
        logger = tools.Logger(config, logdir, config.action_repeat * step)

        agent = Dreamer(
            envs[0].observation_space,
            envs[0].action_space,
            config,
            logger,
            None,
        ).to(config.device)

        agent.requires_grad_(requires_grad=False)

        checkpoint_path = pathlib.Path(config.agent_checkpoint_dir) / "latest.pt"
        print(checkpoint_path)
        
        if checkpoint_path.exists():
            print("Loading checkpoint from", config.agent_checkpoint_dir)
            checkpoint = torch.load(checkpoint_path)
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False

        policy = functools.partial(agent, training=False)

        collect_rollouts(config, envs, policy)
        
    for env in envs:
        try:
            env.close()
        except Exception:
            pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()

    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name]) 

    parser = argparse.ArgumentParser()

    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))

    main(parser.parse_args(remaining))