# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, List, Optional

import numpy as np
from gym import spaces

from mtenv import MTEnv
from mtenv.utils import seeding
from mtenv.utils.types import ActionType, ObsType, StepReturnType

TaskStateType = int


class FiniteMTBanditEnv(MTEnv):
    """Multitask Bandit Env where the task_state is sampled from a finite list of states"""

    def __init__(self, n_tasks: int, n_arms: int):
        super().__init__(
            action_space=spaces.Discrete(n_arms),
            env_observation_space=spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            task_observation_space=spaces.Box(low=0.0, high=1.0, shape=(n_arms,)),
        )
        self.n_arms = n_arms
        self.n_tasks = n_tasks
        self.observation_space["task_obs"].seed(0)
        self.possible_task_observations = np.asarray(
            [self.observation_space["task_obs"].sample() for _ in range(self.n_tasks)]
        )
        # possible_task_observations is assumed to be part of the environment definition ie
        # everytime we instantiate the env, we get the same `possible_task_observations`.
        self._should_reset_env = True

    def reset(self, **kwargs: Any) -> ObsType:
        self.assert_env_seed_is_set()
        self._should_reset_env = False
        return {"env_obs": [0.0], "task_obs": self.task_obs}

    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state` that contains all the information needed to revert to any
        other task. For examples, refer to TBD"""
        self.assert_task_seed_is_set()
        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.

        return self.np_random_task.randint(0, self.n_tasks)  # type: ignore[no-any-return, union-attr]

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.task_state = task_state
        self.task_obs = self.possible_task_observations[task_state]

    def step(self, action: ActionType) -> StepReturnType:
        if self._should_reset_env:
            raise RuntimeError("Call `env.reset()` before calling `env.step()`")
        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        sample = self.np_random_env.rand()  # type: ignore[union-attr]
        reward = 0.0
        if sample < self.task_obs[action]:  # type: ignore[index]
            reward = 1.0

        return (
            {"env_obs": [0.0], "task_obs": self.task_obs},
            reward,
            False,
            {},
        )

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for task information"""
        self.np_random_task, seed = seeding.np_random(seed)
        # in this function, we do not need the self.np_random_task
        return [seed]

    def get_task_state(self) -> TaskStateType:
        """Return all the information needed to execute the current task again.
        For examples, refer to TBD"""
        return self.task_state


def run() -> None:
    env = FiniteMTBanditEnv(n_tasks=10, n_arms=5)
    env.seed(seed=1)
    env.seed_task(seed=2)

    for task in range(3):
        print("=== Task " + str(task % 2))
        env.set_task_state(task % 2)
        print(env.reset())
        for _ in range(5):
            action = env.action_space.sample()
            print(env.step(action))

    new_env = FiniteMTBanditEnv(n_tasks=10, n_arms=5)
    new_env.seed(seed=1)
    new_env.seed_task(seed=2)

    print("=== Executing the current task (from old env) in new env ")

    new_env.set_task_state(task_state=env.get_task_state())
    print(new_env.reset())
    for _ in range(5):
        action = new_env.action_space.sample()
        print(new_env.step(action))


if __name__ == "__main__":
    run()
