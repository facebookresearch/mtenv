# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict

import gym
from gym.core import Env
from gym.envs.registration import register

from mtenv.envs.hipbmdp.wrappers import framestack, sticky_observation


def _build_env(
    domain_name: str,
    task_name: str,
    seed: int = 1,
    xml_file_id: str = "none",
    visualize_reward: bool = True,
    from_pixels: bool = False,
    height: int = 84,
    width: int = 84,
    camera_id: int = 0,
    frame_skip: int = 1,
    environment_kwargs: Any = None,
    episode_length: int = 1000,
) -> Env:
    if xml_file_id is None:
        env_id = "dmc_%s_%s_%s-v1" % (domain_name, task_name, seed)
    else:
        env_id = "dmc_%s_%s_%s_%s-v1" % (domain_name, task_name, xml_file_id, seed)

    if from_pixels:
        assert (
            not visualize_reward
        ), "cannot use visualize reward when learning from pixels"

    # shorten episode length
    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.envs.registry.env_specs:
        register(
            id=env_id,
            entry_point="mtenv.envs.hipbmdp.wrappers.dmc_wrapper:DMCWrapper",
            kwargs={
                "domain_name": domain_name,
                "task_name": task_name,
                "task_kwargs": {"random": seed, "xml_file_id": xml_file_id},
                "environment_kwargs": environment_kwargs,
                "visualize_reward": visualize_reward,
                "from_pixels": from_pixels,
                "height": height,
                "width": width,
                "camera_id": camera_id,
                "frame_skip": frame_skip,
            },
            max_episode_steps=max_episode_steps,
        )
    return gym.make(env_id)


def build_dmc_env(
    domain_name: str,
    task_name: str,
    seed: int,
    xml_file_id: str,
    visualize_reward: bool,
    from_pixels: bool,
    height: int,
    width: int,
    frame_skip: int,
    frame_stack: int,
    sticky_observation_cfg: Dict[str, Any],
) -> Env:
    """Build a single DMC environment as described in
    :cite:`tassa2020dmcontrol`.

    Args:
        domain_name (str): name of the domain.
        task_name (str): name of the task.
        seed (int): environment seed (for reproducibility).
        xml_file_id (str): id of the xml file to use.
        visualize_reward (bool): should visualize reward ?
        from_pixels (bool): return pixel observations?
        height (int): height of pixel frames.
        width (int): width of pixel frames.
        frame_skip (int): should skip frames?
        frame_stack (int): should stack frames together?
        sticky_observation_cfg (Dict[str, Any]): Configuration for using
            sticky observations. It should be a dictionary with three
            keys, `should_use` which specifies if the config should be
            used, `sticky_probability` which specifies the probability of
            choosing a previous task and `last_k` which specifies the
            number of previous frames to choose from.

    Returns:
        Env:
    """
    env = _build_env(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=visualize_reward,
        from_pixels=from_pixels,
        height=height,
        width=width,
        frame_skip=frame_skip,
        xml_file_id=xml_file_id,
    )
    if from_pixels:
        env = framestack.FrameStack(env, k=frame_stack)
    if sticky_observation_cfg and sticky_observation_cfg["should_use"]:
        env = sticky_observation.StickyObservation(  # type: ignore[attr-defined]
            env=env,
            sticky_probability=sticky_observation_cfg["sticky_probability"],
            last_k=sticky_observation_cfg["last_k"],
        )
    return env
