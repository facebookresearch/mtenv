# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Callable, Dict, List

from gym.core import Env

from mtenv import MTEnv
from mtenv.envs.hipbmdp import dmc_env
from mtenv.envs.shared.wrappers.multienv import MultiEnvWrapper

EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int


def build(
    domain_name: str,
    task_name: str,
    seed: int,
    xml_file_ids: List[str],
    visualize_reward: bool,
    from_pixels: bool,
    height: int,
    width: int,
    frame_skip: int,
    frame_stack: int,
    sticky_observation_cfg: Dict[str, Any],
    initial_task_state: int = 1,
) -> MTEnv:
    def get_func_to_make_envs(xml_file_id: str) -> EnvBuilderType:
        def _func() -> Env:
            return dmc_env.build_dmc_env(
                domain_name=domain_name,
                task_name=task_name,
                seed=seed,
                xml_file_id=xml_file_id,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                height=height,
                width=width,
                frame_skip=frame_skip,
                frame_stack=frame_stack,
                sticky_observation_cfg=sticky_observation_cfg,
            )

        return _func

    funcs_to_make_envs = [
        get_func_to_make_envs(xml_file_id=file_id) for file_id in xml_file_ids
    ]

    mtenv = MultiEnvWrapper(
        funcs_to_make_envs=funcs_to_make_envs, initial_task_state=initial_task_state
    )
    return mtenv
