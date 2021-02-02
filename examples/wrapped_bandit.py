# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional

from gym import spaces

from examples.bandit import BanditEnv  # type: ignore[import]
from mtenv.utils import seeding
from mtenv.utils.types import TaskObsType, TaskStateType
from mtenv.wrappers.env_to_mtenv import EnvToMTEnv


class MTBanditWrapper(EnvToMTEnv):
    def set_task_observation(self, task_obs: TaskObsType) -> None:
        self.task_obs = task_obs
        self.env.reward_probability = self.task_obs
        self._is_task_seed_set = False

    def get_task_state(self) -> TaskStateType:
        return self.task_obs

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.task_obs = task_state
        self.env.reward_probability = self.task_obs

    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state` that contains all the information needed to revert to any
        other task. For examples, refer to TBD"""
        return self.observation_space["task_obs"].sample()

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for task information"""
        self._is_task_seed_set = True
        _, seed = seeding.np_random(seed)
        self.observation_space["task_obs"].seed(seed)
        return [seed]

    def assert_task_seed_is_set(self) -> None:
        """Check that the task seed is set."""
        assert self._is_task_seed_set, "please call `seed_task()` first"


def run() -> None:
    n_arms = 5
    env = MTBanditWrapper(
        env=BanditEnv(n_arms),
        task_observation_space=spaces.Box(low=0.0, high=1.0, shape=(n_arms,)),
    )
    env.seed(1)
    env.seed_task(seed=2)
    for task in range(3):
        print("=== task " + str(task))
        env.reset_task_state()
        print(env.reset())
        for _ in range(5):
            action = env.action_space.sample()
            print(env.step(action))
        print(f"reward_probability: {env.unwrapped.reward_probability}")


if __name__ == "__main__":
    run()
