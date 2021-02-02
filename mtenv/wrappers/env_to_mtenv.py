# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper to convert an environment into multitask environment."""
from typing import Any, Dict, List, Optional

from gym.core import Env
from gym.spaces.space import Space

from mtenv import MTEnv
from mtenv.utils import seeding
from mtenv.utils.types import (
    ActionType,
    EnvObsType,
    ObsType,
    StepReturnType,
    TaskObsType,
    TaskStateType,
)


class EnvToMTEnv(MTEnv):
    def __init__(self, env: Env, task_observation_space: Space) -> None:
        """Wrapper to convert an environment into a multitak environment.

        Args:
            env (Env): Environment to wrap over.
            task_observation_space (Space): Task observation space for the
                resulting multitask environment.
        """

        super().__init__(
            action_space=env.action_space,
            env_observation_space=env.observation_space,
            task_observation_space=task_observation_space,
        )

        self.env = env
        self.reward_range = self.env.reward_range
        self.metadata = self.env.metadata

    @property
    def spec(self) -> Any:
        return self.env.spec

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def _make_observation(self, env_obs: EnvObsType) -> ObsType:
        return {"env_obs": env_obs, "task_obs": self.get_task_obs()}

    def get_task_obs(self) -> TaskObsType:
        return self._task_obs

    def get_task_state(self) -> TaskStateType:
        raise NotImplementedError

    def set_task_state(self, task_state: TaskStateType) -> None:
        raise NotImplementedError

    def sample_task_state(self) -> TaskStateType:
        raise NotImplementedError

    def reset(self, **kwargs: Dict[str, Any]) -> ObsType:
        self.assert_env_seed_is_set()
        env_obs = self.env.reset(**kwargs)
        return self._make_observation(env_obs=env_obs)

    def reset_task_state(self) -> None:
        self.set_task_state(task_state=self.sample_task_state())

    def step(self, action: ActionType) -> StepReturnType:
        env_obs, reward, done, info = self.env.step(action)
        return (
            self._make_observation(env_obs=env_obs),
            reward,
            done,
            info,
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random_env, seed = seeding.np_random(seed)
        env_seeds = self.env.seed(seed)
        if isinstance(env_seeds, list):
            return [seed] + env_seeds
        return [seed]

    def render(self, mode: str = "human", **kwargs: Dict[str, Any]) -> Any:
        """Renders the environment."""
        return self.env.render(mode, **kwargs)

    def close(self) -> Any:
        return self.env.close()

    def __str__(self) -> str:
        return f"{type(self).__name__}{self.env}"

    def __repr__(self) -> str:
        return str(self)

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.env, name)
