# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from abc import ABC, abstractmethod
from typing import List, Optional

from gym.core import Env
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.space import Space
from numpy.random import RandomState

from mtenv.utils import seeding
from mtenv.utils.types import (
    ActionType,
    ObsType,
    StepReturnType,
    TaskObsType,
    TaskStateType,
)


class MTEnv(Env, ABC):  # type: ignore[misc]
    def __init__(
        self,
        action_space: Space,
        env_observation_space: Space,
        task_observation_space: Space,
    ) -> None:
        """This abstract base class extends the OpenAI Gym class with
            task information and is the main class for interacting with mult-task environments.

        Args:
            action_space (Space): [description]
            env_observation_space (Space): [description]
            task_observation_space (Space): [description]
        """

        self.action_space = action_space
        self.observation_space: DictSpace = DictSpace(
            spaces={
                "env_obs": env_observation_space,
                "task_obs": task_observation_space,
            }
        )

        self.np_random_env: Optional[RandomState] = None
        self.np_random_task: Optional[RandomState] = None

        self._task_obs: TaskObsType

    @abstractmethod
    def step(self, action: ActionType) -> StepReturnType:
        """Execute the action in the environment and return a tuple of
        `(observation, reward, done, info)`."""
        pass

    def get_task_obs(self) -> TaskObsType:
        """Get the current value of `task_obs` attribute. `task_obs` is
        returned as part of every observation as well. This function is useful
        in cases where the user only wants to access the task_obs and does not
        want to execute an action in the environment or reset the environment."""
        return self._task_obs

    @abstractmethod
    def get_task_state(self) -> TaskStateType:
        """Return all the information needed to execute the current task again.
        For examples, refer to TBD"""
        pass

    @abstractmethod
    def set_task_state(self, task_state: TaskStateType) -> None:
        """task_state contains all the information needed to revert to any
        other task. For examples, refer to TBD"""
        pass

    def assert_env_seed_is_set(self) -> None:
        """Check that the env seed is set."""
        assert self.np_random_env is not None, "please call `seed()` first"

    def assert_task_seed_is_set(self) -> None:
        """Check that the task seed is set."""
        assert self.np_random_task is not None, "please call `seed_task()` first"

    @abstractmethod
    def reset(self) -> ObsType:
        """Reset the environment to a start state and return the observation
        corresponding to that state.

        The subclasses extending this class are recommended to ensure
        that the seed is set (by calling `seed(int)`) before invoking this
        method. It can be done by eg:

        ```self.assert_env_seed_is_set()```

        """
        pass

    @abstractmethod
    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state` that contains all the information needed to revert to any
        other task. For examples, refer to TBD.

        The subclasses extending this class are recommended to ensure
        that the seed is set (by calling `seed(int)`) before invoking this
        method. It can be done by eg:

        ```self.assert_task_seed_is_set()```
        """
        pass

    def reset_task_state(self) -> None:
        """Sample a new task_state and set that as the new task_state"""
        self.set_task_state(task_state=self.sample_task_state())

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for environment observations"""
        self.np_random_env, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        return [seed]

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for task information"""
        self.np_random_task, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        self.observation_space["task_obs"].seed(seed)
        return [seed]
