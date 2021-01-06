# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Core API of MultiTask Environments for Reinforcement Learning."""
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
        """Main class for multitask RL Environments.

        This abstract class extends the OpenAI Gym environment and adds
        support for return the task-specific information from the environment.
        The observation returned from the single task environments is
        encoded as `env_obs` (environment observation) while the task
        specific observation is encoded as the `task_obs` (task observation).
        The observation returned by `mtenv` is a dictionary of `env_obs` and
        `task_obs`.  Since this class extends the OpenAI gym, the `mtenv`
        API looks similar to the gym API.

        .. code-block:: python

            import mtenv
            env = mtenv.make('xxx')
            env.reset()

        Any multitask RL environment class should extend/implement this class.

        Args:
            action_space (Space)
            env_observation_space (Space)
            task_observation_space (Space)
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
        """Execute the action in the environment.

        Args:
            action (ActionType)

        Returns:
            StepReturnType: Tuple of `multitask observation`, `reward`,
            `done`, and `info`. For more information on `multitask observation`
            returned by the environment, refer :ref:`multitask_observation`.
        """
        pass

    def get_task_obs(self) -> TaskObsType:
        """Get the current value of task observation.

        Environment returns task observation everytime we call `step` or
        `reset`. This function is useful when the user wants to access the
        task observation without acting in (or resetting) the environment.

        Returns:
            TaskObsType:
        """
        return self._task_obs

    @abstractmethod
    def get_task_state(self) -> TaskStateType:
        """Return all the information needed to execute the current task
        again.

        This function is useful when we want to set the environment to a
        previous task.

        Returns:
            TaskStateType: For more information on `task_state`, refer :ref:`task_state`.
        """
        pass

    @abstractmethod
    def set_task_state(self, task_state: TaskStateType) -> None:
        """Reset the environment to a particular task.

        `task_state` contains all the information that the environment
        needs to switch to any other task.

        Args:
            task_state (TaskStateType): For more information on `task_state`,
                refer :ref:`task_state`.
        """
        pass

    def assert_env_seed_is_set(self) -> None:
        """Check that seed (for the environment) is set.

        `reset` function should invoke this function before resetting the
        environment (for reproducibility).

        """
        assert self.np_random_env is not None, "please call `seed()` first"

    def assert_task_seed_is_set(self) -> None:
        """Check that seed (for the task) is set.

        `sample_task_state` function should invoke this function before
        sampling a new task state (for reproducibility).

        """
        assert self.np_random_task is not None, "please call `seed_task()` first"

    @abstractmethod
    def reset(self) -> ObsType:
        """Reset the environment to some initial state and return the
        observation in the new state.

        The subclasses, extending this class, should ensure that the
        environment seed is set (by calling `seed(int)`) before invoking this
        method (for reproducibility). It can be done by invoking
        `self.assert_env_seed_is_set()`.

        Returns:
            ObsType: For more information on `multitask observation`
            returned by the environment, refer :ref:`multitask_observation`.
        """
        pass

    @abstractmethod
    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state`.

        `task_state` contains all the information that the environment
        needs to switch to any other task.

        The subclasses, extending this class, should ensure that the task
        seed is set (by calling `seed(int)`) before invoking this
        method (for reproducibility). It can be done by invoking
        `self.assert_task_seed_is_set()`.

        Returns:
            TaskStateType: For more information on `task_state`,
            refer :ref:`task_state`.
        """
        pass

    def reset_task_state(self) -> None:
        """Sample a new task_state and set the environment to that `task_state`.

        For more information on `task_state`, refer :ref:`task_state`.
        """
        self.set_task_state(task_state=self.sample_task_state())

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for the environment's random number generator.

        Invoke `seed_task` to set the seed for the task's
        random number generator.

        Args:
            seed (Optional[int], optional): Defaults to None.

        Returns:
            List[int]: Returns the list of seeds used in the environment's
            random number generator. The first value in the list should be
            the seed that should be passed to this method for reproducibility.
        """
        self.np_random_env, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        return [seed]

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for the task's random number generator.

        Invoke `seed` to set the seed for the environment's
        random number generator.

        Args:
            seed (Optional[int], optional): Defaults to None.

        Returns:
            List[int]: Returns the list of seeds used in the task's
            random number generator. The first value in the list should be
            the seed that should be passed to this method for reproducibility.
        """
        self.np_random_task, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        self.observation_space["task_obs"].seed(seed)
        return [seed]
