import argparse
import functools
import os
import pathlib
import sys
import torch
import ruamel.yaml as yaml
from datetime import datetime

import tools
from parallel import Parallel, Damy

from expr import LS_Imagine, make_env, make_dataset

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()

def main(config):

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser() 
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
    config.logdir = logdir

    config.evaldir = config.evaldir or logdir / "eval_eps"
    logdir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)

    step = 0
    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_evaldir: # False
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir

    eval_eps = tools.load_episodes(directory, limit=1)

    make = lambda mode, id: make_env(config, mode, id)
    suite, task = config.task.split("_", 1)

    from envs.tasks import get_specs

    kwargs=dict(
            # log_dir=log_dir,
            target_item=config.target_item
        )
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict

    config.episode_max_steps = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['max_steps'] = task_specs['terminal_specs']['max_steps']
    task_specs['concentration_specs']['gaussian_reward_weight'] = config.gaussian_reward_weight
    task_specs['concentration_specs']['gaussian_sigma_weight'] = config.gaussian_sigma_weight
    task_specs['clip_specs']['target_object'] = task_specs['success_specs']['all']['item']['type'] if 'all' in task_specs['success_specs'] else task_specs['success_specs']['any']['item']['type']

    eval_envs = [make("eval", i) for i in range(config.envs)]
    if config.parallel:
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        eval_envs = [Damy(env) for env in eval_envs]
    acts = eval_envs[0].action_space

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step_calculator = tools.ScoreStorage(max_steps=config.episode_max_steps)

    state = None

    print("Start evaluation.")
    eval_dataset = make_dataset(eval_eps, config)
    agent = LS_Imagine(
        eval_envs[0].observation_space,
        eval_envs[0].action_space,
        config,
        logger,
        eval_dataset,
    ).to(config.device)

    checkpoint_path = pathlib.Path(config.agent_checkpoint_dir) / "latest.pt"
    print(checkpoint_path)
    
    if checkpoint_path.exists():
        print("Loading checkpoint from", config.agent_checkpoint_dir)
        checkpoint = torch.load(checkpoint_path)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    else:
        raise ValueError("no checkpoint found")
    
    if config.eval_episode_num > 0:
        eval_policy = functools.partial(agent, training=False) 
        tools.simulate(
            eval_policy,
            eval_envs,
            eval_eps,
            config.evaldir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            is_eval=True,
            episodes=config.eval_episode_num,
            is_training=False,
        )
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_openl", to_np(video_pred))

    for env in eval_envs:
        try:
            env.close()
        except Exception:
            pass

    logger.finish()


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

