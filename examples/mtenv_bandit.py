# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from gym import spaces

from mtenv import MTEnv
from mtenv.utils.types import ActionType, ObsType, StepReturnType, TaskStateType


class MTBanditEnv(MTEnv):
    """Multitask Bandit Env based on Ludovic's implementation.
    Note that this env has an explicit notion of task"""

    def __init__(self, n_arms: int):
        super().__init__(
            action_space=spaces.Discrete(n_arms),
            env_observation_space=spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            task_observation_space=spaces.Box(low=0.0, high=1.0, shape=(n_arms,)),
        )
        self.n_arms = n_arms
        self._should_reset_env = True

    def reset(self) -> ObsType:
        self.assert_env_seed_is_set()
        self._should_reset_env = False
        return {"env_obs": [0.0], "task_obs": self.task_observation}

    def sample_task_state(self) -> TaskStateType:
        self.assert_task_seed_is_set()
        return self.observation_space["task_obs"].sample()

    def get_task_state(self) -> TaskStateType:
        return self.task_observation

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.task_observation = task_state

    def step(self, action: ActionType) -> StepReturnType:
        if self._should_reset_env:
            raise RuntimeError("Call `env.reset()` before calling `env.step()`")

        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        sample = self.np_random_env.rand()  # type: ignore[union-attr]
        reward = 0.0
        if sample < self.task_observation[action]:
            reward = 1.0

        return (
            {"env_obs": [0.0], "task_obs": self.task_observation},
            reward,
            False,
            {},
        )


def run() -> None:
    env = MTBanditEnv(5)
    env.seed(seed=1)
    env.seed_task(seed=2)

    for task in range(3):
        print("=== Task " + str(task))
        env.reset_task_state()
        print(env.reset())
        for _ in range(5):
            action = env.action_space.sample()
            print(env.step(action))


if __name__ == "__main__":
    run()
