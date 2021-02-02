# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from copy import deepcopy

from mtenv.envs.registration import register

# Control Task
# ----------------------------------------

register(
    id="MT-CartPole-v0",
    entry_point="mtenv.envs.control.cartpole:MTCartPole",
    test_kwargs={
        # "valid_env_kwargs": [],
        "invalid_env_kwargs": [],
    },
)


register(
    id="MT-TabularMDP-v0",
    entry_point="mtenv.envs.tabular_mdp.tmdp:UniformTMDP",
    kwargs={"n_states": 4, "n_actions": 5},
    test_kwargs={
        "valid_env_kwargs": [{"n_states": 3, "n_actions": 2}],
        "invalid_env_kwargs": [],
    },
)

register(
    id="MT-Acrobat-v0",
    entry_point="mtenv.envs.control.acrobot:MTAcrobot",
    test_kwargs={
        # "valid_env_kwargs": [],
        "invalid_env_kwargs": [],
    },
)

register(
    id="MT-TwoGoalMaze-v0",
    entry_point="mtenv.envs.mpte.two_goal_maze_env:build_two_goal_maze_env",
    kwargs={"size_x": 3, "size_y": 3, "task_seed": 169, "n_tasks": 100},
    test_kwargs={
        # "valid_env_kwargs": [],
        "invalid_env_kwargs": [],
    },
)


# remove it before making the repo public.
default_kwargs = {
    "seed": 1,
    "visualize_reward": False,
    "from_pixels": True,
    "height": 84,
    "width": 84,
    "frame_skip": 2,
    "frame_stack": 3,
    "sticky_observation_cfg": {},
    "initial_task_state": 1,
}

for domain_name, task_name, prefix in [
    ("finger", "spin", "size"),
    ("cheetah", "run", "torso_length"),
    ("walker", "walk", "friction"),
    ("walker", "walk", "len"),
]:
    file_ids = list(range(1, 11))
    kwargs = deepcopy(default_kwargs)
    kwargs["domain_name"] = domain_name
    kwargs["task_name"] = task_name
    kwargs["xml_file_ids"] = [f"{prefix}_{i}" for i in file_ids]
    register(
        id=f"MT-HiPBMDP-{domain_name.capitalize()}-{task_name.capitalize()}-vary-{prefix.replace('_', '-')}-v0",
        entry_point="mtenv.envs.hipbmdp.env:build",
        kwargs=kwargs,
        test_kwargs={
            # "valid_env_kwargs": [],
            # "invalid_env_kwargs": [],
        },
    )


default_kwargs = {
    "benchmark": None,
    "benchmark_name": "MT10",
    "env_id_to_task_map": None,
    "should_perform_reward_normalization": True,
    "num_copies_per_env": 1,
    "initial_task_state": 1,
}

for benchmark_name in [("MT10"), ("MT50")]:
    kwargs = deepcopy(default_kwargs)
    kwargs["benchmark_name"] = benchmark_name
    register(
        id=f"MT-MetaWorld-{benchmark_name}-v0",
        entry_point="mtenv.envs.metaworld.env:build",
        kwargs=kwargs,
        test_kwargs={
            # "valid_env_kwargs": [],
            # "invalid_env_kwargs": [],
        },
    )

kwargs = {
    "benchmark": None,
    "benchmark_name": "MT1",
    "env_id_to_task_map": None,
    "should_perform_reward_normalization": True,
    "task_name": "pick-place-v1",
    "num_copies_per_env": 1,
    "initial_task_state": 0,
}
register(
    id=f'MT-MetaWorld-{kwargs["benchmark_name"]}-v0',
    entry_point="mtenv.envs.metaworld.env:build",
    kwargs=kwargs,
    test_kwargs={
        # "valid_env_kwargs": [],
        # "invalid_env_kwargs": [],
    },
)
