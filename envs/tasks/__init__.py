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

def make(task: str, **kwargs):
    task_id, task_specs, sim_specs = get_specs(task, **kwargs)  # Note: additional kwargs end up in task_specs dict
    env = make_minedojo(task_id, task_specs, sim_specs)

    return env
