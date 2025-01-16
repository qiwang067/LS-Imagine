# How to Set Up Tasks in MineDojo

We have predefined the task settings mentioned in the [paper](https://arxiv.org/pdf/2410.03618) in the `./envs/tasks/task_specs.yaml` file. Each task includes configurations for three stages: data collection, world model and behavior learning, and testing. For example, the task ***Harvest log in plains*** is defined as `collect_rollouts_harvest_log_in_plains`, `harvest_log_in_plains`, and `test_harvest_log_in_plains` for these three stages, respectively. 

This document introduces the meaning and configuration guidelines for each field in the task settings, enabling you to define your custom tasks by following these examples.

## Field Descriptions and Guidelines

- **`task_id`**: Generally set to `harvest`. For more details, refer to [Task Customization](https://docs.minedojo.org/sections/customization/task.html#task-customization).
- **`sim`**: Typically set to `minedojo`.
- **`fast_reset`**: For more information, refer to [Reset Mode](https://docs.minedojo.org/sections/customization/task.html#reset-mode).  
  - During world model and behavior learning, `fast_reset` is often set to `5` or `10` to stabilize the training process.  
  - During data collection and testing, `fast_reset` is usually set to `0` to collect more diverse data.

- **`sim_specs`**: Details of the environment setup. See [Task Customization](https://docs.minedojo.org/sections/customization/task.html#task-customization) for more information.
  - **`target_names`**: Specifies the names of the target items for `harvest` tasks.
  - **`target_quantities`**: Specifies the quantities of the target items for `harvest` tasks.
  - **`specified_biome`**: Defines the initial biome of the environment. See [Specified Biome](https://docs.minedojo.org/sections/customization/sim.html#specified-biome).
  - **`break_speed_multiplier`**: Adjusts the breaking speed of attacks. See [Set Breaking Speed](https://docs.minedojo.org/sections/customization/sim.html#set-breaking-speed). Commonly set to `100` to simplify training.
  - **`initial_inventory`**: Specifies the initial inventory setup. See [Initial Inventory](https://docs.minedojo.org/sections/customization/sim.html#initial-inventory).
  - **`initial_mobs`**: Specifies the initial mobs in the environment. See [Spawn Mobs](https://docs.minedojo.org/sections/customization/sim.html#spawn-mobs).

- **`clip_specs`**: Configuration for using `MineCLIP` to calculate intrinsic rewards.
  - **`prompts`**: Short textual descriptions of the task, specified as `list[str]`.

- **`concentration_specs`**: Configuration for calculating intrinsic rewards based on affordance maps and jumping flags.
  - **`prompts`**: Short textual descriptions of the task, specified as `list[str]`.
  - **`unet_checkpoint_dir`**: Path to the multimodal U-Net weight files.

- **`reward_specs`**: Specifies rewards for collecting different items.

- **`success_specs`**: Configures additional success criteria and actions upon success.
  - **`reward`**: Additional reward upon success.
  - **`any`**: Specifies that the task is considered successful if any of the listed conditions are met.
  - **`all`**: Specifies that the task is considered successful if all listed conditions are met.

- **`terminal_specs`**: Configures conditions for ending an episode.
  - **`max_steps`**: Specifies the maximum number of steps per episode.

- **`screenshot_specs`**: Configuration for saving observation images.
  - **`reset_flag`**: Whether to save images during `reset`. (Images during reset are often darker.)
  - **`step_flag`**: Whether to save images during each `step`.
  - **`save_freq`**: Interval between saved images on average.

- **`LS_Imagine_specs`**: Settings for integrating the LS-Imagine algorithm with the MineDojo environment.
  - **`repeat`**: Number of times a single action is repeated. Typically set to `1`.