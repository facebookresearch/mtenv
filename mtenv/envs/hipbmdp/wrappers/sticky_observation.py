# type: ignore

import random
from collections import deque

import gym


class StickyObservation(gym.Wrapper):
    """Env wrapper that returns a previous observation with probability p
    and the current observation with a probability 1-p. last_k previous
    observations are stored."""

    def __init__(self, env, sticky_probability: float, last_k: int):
        gym.Wrapper.__init__(self, env)
        if 1 >= sticky_probability >= 0:
            self._sticky_probability = sticky_probability
        else:
            raise ValueError(
                f"sticky_probability = {sticky_probability} is not in the interval [0, 1]."
            )
        self._last_k = last_k + 1
        self._observations = deque([], maxlen=self._last_k)
        self.observation_space = env.observation_space
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._last_k):
            self._observations.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._observations.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._observations) == self._last_k
        should_choose_old_observation = random.random() < self._sticky_probability
        if should_choose_old_observation:
            index = random.randint(0, self._last_k - 2)
            return self._observations[index]
        else:
            return self._observations[-1]
