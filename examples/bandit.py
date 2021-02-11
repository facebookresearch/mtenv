# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List, Optional, Tuple

import numpy as np
from gym import spaces
from gym.core import Env

from mtenv.utils import seeding
from mtenv.utils.types import ActionType, DoneType, EnvObsType, InfoType, RewardType

StepReturnType = Tuple[EnvObsType, RewardType, DoneType, InfoType]


class BanditEnv(Env):  # type: ignore[misc]
    # Class cannot subclass 'Env' (has type 'Any')

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.reward_probability = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_arms,)
        ).sample()

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random_env, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        return [seed]

    def reset(self) -> EnvObsType:
        return np.asarray([0.0])

    def step(self, action: ActionType) -> StepReturnType:
        sample = self.np_random_env.rand()
        reward = 0.0
        if sample < self.reward_probability[action]:
            reward = 1.0

        return np.asarray([0.0]), reward, False, {}


def run() -> None:
    env = BanditEnv(5)
    env.seed(seed=5)
    for episode in range(3):
        print("=== episode " + str(episode))
        print(env.reset())
        for _ in range(5):
            action = env.action_space.sample()
            print(env.step(action))


if __name__ == "__main__":
    run()
