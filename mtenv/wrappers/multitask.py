# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from numpy.random import RandomState

from mtenv.core import MTEnv
from mtenv.utils import seeding
from mtenv.utils.types import (
    ActionType,
    EnvObsType,
    StepReturnType,
    TaskObsType,
    TaskStateType,
)


class MultiTask(MTEnv):
    """
    A wrapper class that wraps over a multi-task environment.
    """

    def __init__(self, env: MTEnv):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.np_random_env: Optional[RandomState] = None
        self.np_random_task: Optional[RandomState] = None

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

    def reset(self) -> EnvObsType:
        return self.env.reset()

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
