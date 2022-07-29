# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper to change the behaviour of an existing multitask environment."""

from typing import Any, Dict, List, Optional, Tuple, Union

from mtenv import MTEnv
from mtenv.utils import seeding
from mtenv.utils.types import (
    ActionType,
    InfoType,
    ObsType,
    StepReturnType,
    TaskObsType,
    TaskStateType,
)


class MultiTask(MTEnv):
    def __init__(self, env: MTEnv):
        """Wrapper to change the behaviour of an existing multitask environment

        Args:
            env (MTEnv): Multitask environment to wrap over.
        """
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def step(self, action: ActionType) -> StepReturnType:
        return self.env.step(action)

    def get_task_obs(self) -> TaskObsType:
        return self.env.get_task_obs()

    def get_task_state(self) -> TaskStateType:
        return self.env.get_task_state()

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.env.set_task_state(task_state)

    def assert_env_seed_is_set(self) -> None:
        """Check that the env seed is set."""
        assert self.np_random_env is not None, "please call `seed()` first"
        self.env.assert_env_seed_is_set()

    def assert_task_seed_is_set(self) -> None:
        """Check that the task seed is set."""
        assert self.np_random_task is not None, "please call `seed_task()` first"
        self.env.assert_task_seed_is_set()

    def reset(  # type: ignore[override]
        self,
        *,
        env_seed: Optional[int] = None,
        task_seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[Dict[str, Any]] = None,
    ) -> Union[ObsType, Tuple[ObsType, InfoType]]:
        return self.env.reset(
            env_seed=env_seed,
            task_seed=task_seed,
            return_info=return_info,
            options=options,
        )

    def sample_task_state(self) -> TaskStateType:
        return self.env.sample_task_state()

    def reset_task_state(self) -> None:
        self.env.reset_task_state()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random_env, seed = seeding.np_random(seed)
        return [seed] + self.env.seed(seed)

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        self.np_random_task, seed = seeding.np_random(seed)
        return [seed] + self.env.seed_task(seed)
