import argparse
import functools
import os
import pathlib
import sys
import torch
import numpy as np
import ruamel.yaml as yaml
from torch import nn
from torch import distributions as torchd
from datetime import datetime

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

os.environ["MUJOCO_GL"] = "osmesa"
sys.path.append(str(pathlib.Path(__file__).parent))

to_np = lambda x: x.detach().cpu().numpy()


class LS_Imagine(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(LS_Imagine, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, post_zoomed, context, mets = self._wm._train(data)
        metrics.update(mets)
        # start = (post, post_zoomed)

        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()

        intrinsic = lambda f, s, a: self._wm.heads["intrinsic"](
            self._wm.dynamics.get_feat(s)
        ).mode() 

        jumping_steps = lambda f, s, a: self._wm.heads["jumping_steps"](
            f
        ).mean().clamp_min(1).int()

        accumulated_reward = lambda f, s, a: self._wm.heads["accumulated_reward"](
            f
        ).mode()

        jump_indicator = lambda s: self._wm.heads["jump"](
            self._wm.dynamics.get_feat(s)
        ).mean

        is_end = lambda s: self._wm.heads["end"](
            self._wm.dynamics.get_feat(s)
        ).mean

        metrics.update(self._task_behavior._train(post, post_zoomed, reward, intrinsic, jumping_steps, accumulated_reward, jump_indicator, is_end)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(post, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id):
    suite, task = config.task.split("_", 1)
    if suite == "minedojo":
        import envs.minedojo as minedojo
        log_dir = os.path.join(config.results_dir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        kwargs=dict(
                log_dir=log_dir,
                target_item=config.target_item
            )
        env = minedojo.make_env(task, **kwargs)
        env = wrappers.OneHotAction(env)

    else:
        raise NotImplementedError(suite)
    
    # env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    env = wrappers.RewardObs(env)

    return env


def main(config): # config is namespace

    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    logdir = pathlib.Path(config.logdir).expanduser() 
    logdir = logdir / config.task
    logdir = logdir / 'seed_{}'.format(config.seed)
    timestamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    logdir = logdir / timestamp
    config.logdir = logdir
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat 
    config.log_every //= config.action_repeat 
    config.time_limit //= config.action_repeat
    logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    
    logger = tools.Logger(config, logdir, config.action_repeat * step)

    if config.offline_traindir: # False
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir

    train_eps = tools.load_episodes(directory, limit=config.dataset_size)

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
    
    train_envs = [make("train", i) for i in range(config.envs)]
    eval_envs = [make("eval", i) for i in range(config.envs)]

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space

    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    step_calculator = tools.ScoreStorage(max_steps=config.episode_max_steps)

    state = None

    if not config.offline_traindir: 
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
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

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=prefill,
            is_training=False,
        )

        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = LS_Imagine(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)

    agent.requires_grad_(requires_grad=False)
    
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    
    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every: 
        logger.write()
        
        if config.eval_episode_num > 0:
            print("Start evaluation.")
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

        print("Start training.")

        state = tools.simulate(
            agent, # LS_Imagine
            train_envs, 
            train_eps,
            config.traindir,
            logger,
            step_calculator,
            config.episode_max_steps,
            config.discount,
            limit=config.dataset_size,
            steps=config.eval_every, 
            state=state,
            is_training=True,
        )

        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        
        torch.save(items_to_save, logdir / "latest.pt")

    for env in train_envs + eval_envs:
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
