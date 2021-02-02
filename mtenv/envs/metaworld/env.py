# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import metaworld
from gym.core import Env

from mtenv import MTEnv
from mtenv.envs.metaworld.wrappers.normalized_env import (  # type: ignore[attr-defined]
    NormalizedEnvWrapper,
)
from mtenv.envs.shared.wrappers.multienv import MultiEnvWrapper

EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int
EnvIdToTaskMapType = Dict[str, metaworld.Task]


class MetaWorldMTWrapper(MultiEnvWrapper):
    def __init__(
        self,
        funcs_to_make_envs: List[EnvBuilderType],
        initial_task_state: TaskStateType,
        env_id_to_task_map: EnvIdToTaskMapType,
    ) -> None:
        super().__init__(
            funcs_to_make_envs=funcs_to_make_envs,
            initial_task_state=initial_task_state,
        )
        self.env_id_to_task_map = env_id_to_task_map


def get_list_of_func_to_make_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:
    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def get_func_to_make_envs(env_id: str):
        current_benchmark = benchmark

        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    if should_perform_reward_normalization:
                        env = NormalizedEnvWrapper(env, normalize_reward=True)
                    return env

        return _make_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    funcs_to_make_envs = [get_func_to_make_envs(env_id) for env_id in env_id_list]

    return funcs_to_make_envs, env_id_to_task_map


def build(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
    initial_task_state: int = 1,
) -> MTEnv:
    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_envs(
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        env_id_to_task_map=env_id_to_task_map,
        should_perform_reward_normalization=should_perform_reward_normalization,
        task_name=task_name,
        num_copies_per_env=num_copies_per_env,
    )

    assert env_id_to_task_map is not None

    mtenv = MetaWorldMTWrapper(
        funcs_to_make_envs=funcs_to_make_envs,
        initial_task_state=initial_task_state,
        env_id_to_task_map=env_id_to_task_map,
    )
    return mtenv
