# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Callable, List, Optional

from gym.core import Env
from gym.spaces.discrete import Discrete as DiscreteSpace

from mtenv import MTEnv
from mtenv.utils import seeding
from mtenv.utils.types import ActionType, EnvObsType, ObsType, StepReturnType

EnvBuilderType = Callable[[], Env]
TaskStateType = int
TaskObsType = int


class MultiEnvWrapper(MTEnv):
    def __init__(
        self,
        funcs_to_make_envs: List[EnvBuilderType],
        initial_task_state: TaskStateType,
    ) -> None:
        self._num_tasks = len(funcs_to_make_envs)
        self._funcs_to_make_envs = funcs_to_make_envs
        self._envs = [None for _ in range(self._num_tasks)]
        self._envs[initial_task_state] = funcs_to_make_envs[initial_task_state]()
        self.env: Env = self._envs[initial_task_state]
        super().__init__(
            action_space=self.env.action_space,
            env_observation_space=self.env.observation_space,
            task_observation_space=DiscreteSpace(n=self._num_tasks),
        )
        self.task_obs: TaskObsType = initial_task_state

    def make_observation(self, env_obs: EnvObsType) -> ObsType:
        return {
            "env_obs": env_obs,
            "task_obs": self.task_obs,
        }

    def step(self, action: ActionType) -> StepReturnType:
        env_obs, reward, done, info = self.env.step(action)
        return self.make_observation(env_obs=env_obs), reward, done, info

    def get_task_state(self) -> TaskStateType:
        return self.task_obs

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.task_obs = task_state
        if self._envs[task_state] is None:
            self._envs[task_state] = self._funcs_to_make_envs[task_state]()
        self.env = self._envs[task_state]

    def assert_env_seed_is_set(self) -> None:
        """The seed is set during the call to the constructor of self.env"""
        pass

    def assert_task_seed_is_set(self) -> None:
        """Check that the task seed is set."""
        assert self.np_random_task is not None, "please call `seed_task()` first"

    def reset(self) -> ObsType:
        return self.make_observation(env_obs=self.env.reset())

    def sample_task_state(self) -> TaskStateType:
        self.assert_task_seed_is_set()
        task_state = self.np_random_task.randint(self._num_tasks)  # type: ignore[union-attr]
        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        assert isinstance(task_state, int)
        return task_state

    def reset_task_state(self) -> None:
        """Sample a new task_state and set that as the new task_state"""
        self.set_task_state(task_state=self.sample_task_state())

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for environment observations"""
        self.np_random_env, seed = seeding.np_random(seed)
        env_seeds = self.env.seed(seed)
        if isinstance(env_seeds, list):
            return [seed] + env_seeds
        return [seed]
