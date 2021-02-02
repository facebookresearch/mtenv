# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List, Optional

from gym.core import Env
from gym.spaces.space import Space

from mtenv.core import MTEnv
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
        return {"env_obs": env_obs, "task_obs": self.task_obs}

    def get_task_obs(self) -> TaskObsType:
        """Get the current value of `task_obs` attribute. `task_obs` is
        return as part of every observation as well. This function is useful
        in cases where the user only wants to access the task_obs and do not
        want to execute an action in the environment or reset the environment."""
        return self.task_obs

    def get_task_state(self) -> TaskStateType:
        """Return all the information needed to execute the current task again.
        For examples, refer to TBD"""
        raise NotImplementedError

    def set_task_state(self, task_state: TaskStateType) -> None:
        """task_state contains all the information needed to revert to any
        other task. For examples, refer to TBD"""
        raise NotImplementedError

    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state` that contains all the information needed to revert to any
        other task. For examples, refer to TBD"""
        raise NotImplementedError

    def reset(self, **kwargs: Dict[str, Any]) -> ObsType:
        self.assert_env_seed_is_set()
        env_obs = self.env.reset(**kwargs)
        return self._make_observation(env_obs=env_obs)

    def reset_task_state(self) -> None:
        """Sample a new task_state and set that as the new task_state"""
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
        """Set the seed for environment observations"""
        self.np_random_env, seed = seeding.np_random(seed)
        env_seeds = self.env.seed(seed)
        if isinstance(env_seeds, list):
            return [seed] + env_seeds
        return [seed]

    # these last 6 functions are copy pasted from open-ai gym. could that be a problem?
    def render(self, mode: str = "human", **kwargs: Dict[str, Any]) -> Any:
        return self.env.render(mode, **kwargs)

    def close(self) -> Any:
        return self.env.close()

    def __str__(self) -> str:
        return "<{}{}>".format(type(self).__name__, self.env)

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
