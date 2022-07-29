# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper to stack observations for single task environments."""

from collections import deque
from typing import Any, Dict, Tuple

import gym
import numpy as np

from mtenv.utils.types import ActionType, EnvStepReturnType


class FrameStack(gym.Wrapper):  # type: ignore[type-arg]
    # Mypy error: Class cannot subclass 'Wrapper' (has type 'Any')  [misc]

    def __init__(self, env: gym.core.Env, k: int):
        """Wrapper to stack observations for single task environments.

        Args:
            env (gym.core.Env): Single Task Environment
            k (int): number of frames to stack.
        """
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames: deque = deque([], maxlen=k)
        shp = env.observation_space.shape
        obs_space_dtype = env.observation_space.dtype
        assert obs_space_dtype is not None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),  # type: ignore[index]
            dtype=obs_space_dtype,  # type: ignore[arg-type]
        )
        self._max_episode_steps = env._max_episode_steps  # type: ignore[attr-defined]

    def reset(self, **kwargs: Dict[str, Any]) -> np.ndarray:  # type: ignore[override]
        obs = self.env.reset(**kwargs)  # type: ignore[arg-type]
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action: ActionType) -> EnvStepReturnType:
        obs, *reward_done_info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), *reward_done_info  # type: ignore[return-value]

    def _get_obs(self) -> np.ndarray:
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
