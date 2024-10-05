from omegaconf import OmegaConf
from envs.tasks.minedojo import make_minedojo


CUTOM_TASK_SPECS = OmegaConf.to_container(OmegaConf.load("envs/tasks/task_specs.yaml"))


def get_specs(task, **kwargs):
    # Get task data and task id
    if task in CUTOM_TASK_SPECS:
        yaml_specs = CUTOM_TASK_SPECS[task].copy()
        task_id = yaml_specs.pop("task_id", task)
        assert "sim" in yaml_specs, "task_specs.yaml must define sim attribute"
    else:
        yaml_specs = dict()
        task_id = task

    if "target_item" in kwargs and "base_task" in task:
        yaml_specs["clip_specs"]["prompts"].append("Obtain " + kwargs["target_item"])
        if "bertscore_specs" in yaml_specs:
            yaml_specs["bertscore_specs"]["prompts"].append("Obtain " + kwargs["target_item"])
        if "concentration_specs" in yaml_specs:
            if kwargs["target_item"] == "log":
                yaml_specs["concentration_specs"]["prompts"].append("cut down trees")
            elif kwargs["target_item"] == "cobblestone":
                # yaml_specs["concentration_specs"]["prompts"].append("mine stones")
                yaml_specs["concentration_specs"]["prompts"].append("dig dirt")
                # yaml_specs["concentration_specs"]["prompts"].append("destroy the grass")
            yaml_specs["concentration_specs"]["prompts"].append("Obtain " + kwargs["target_item"])
        yaml_specs["reward_specs"]["item_rewards"][kwargs["target_item"]] = dict(reward=1)
        yaml_specs["success_specs"]["all"]["item"]["type"] = kwargs["target_item"]

    # Get minedojo specs
    sim_specs = yaml_specs.pop("sim_specs", dict())

    # Get our task specs
    task_specs = dict(
        clip=False,
        fake_clip=False,
        fake_dreamer=False,
        subgoals=False,
    )
    task_specs.update(**yaml_specs)
    task_specs.update(**kwargs)
    assert not (task_specs["clip"] and task_specs["fake_clip"]), "Can only use one reward shaper"

    return task_id, task_specs, sim_specs

import numpy as np
import gym

def make(task: str, **kwargs):
    # Get our custom task specs
    print("task:", task)
    print("kwargs:", kwargs)
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict
    #
    #
        #
        #
        #
            # yaml_specs["clip_specs"]["prompts"].append("collect " + kwargs["target_item"])
            # yaml_specs["reward_specs"]["item_rewards"][kwargs["target_item"]] = dict(reward=1)
            # yaml_specs["success_specs"]["all"]["item"]["type"] = kwargs["target_item"]
    #
    
    print("task_id:", task_id) # creative
    print("task_specs:", task_specs)
    print("sim_specs:", sim_specs)

    # Make minedojo env
    env = make_minedojo(task_id, task_specs, sim_specs)

    '''
    if 'concentration_specs' in task_specs and task_specs['concentration_specs'].get('fusion', True):
        original_obs_space = env.observation_space
        new_shape = (*original_obs_space.shape[0:2], 4)  #
        low = np.concatenate((original_obs_space.low.astype(np.float32), np.zeros((128, 128, 1), dtype=np.float32)), axis=2)
        high = np.concatenate((original_obs_space.high.astype(np.float32), np.full((128, 128, 1), 255, dtype=np.float32)), axis=2)
        #
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    '''
    
    return env
