# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Tuple

import gym
import numpy as np

from mtenv import MTEnv
from mtenv.utils.types import (
    DoneType,
    EnvObsType,
    InfoType,
    ObsType,
    RewardType,
    StepReturnType,
)

StepReturnTypeSingleEnv = Tuple[EnvObsType, RewardType, DoneType, InfoType]


def validate_obs_type(obs: ObsType):
    assert isinstance(obs, dict)
    assert "env_obs" in obs
    assert "task_obs" in obs


def validate_step_return_type(step_return: StepReturnType):
    obs, reward, done, info = step_return
    validate_obs_type(obs)
    assert isinstance(reward, (float, int))
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def valiate_obs_type_single_env(obs: EnvObsType):
    assert isinstance(obs, np.ndarray)


def validate_step_return_type_single_env(step_return: StepReturnType):
    obs, reward, done, info = step_return
    valiate_obs_type_single_env(obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def validate_mtenv(env: MTEnv) -> None:
    env.seed(5)
    env.assert_env_seed_is_set()
    env.seed_task(15)
    env.assert_task_seed_is_set()
    for _env_index in range(10):
        env.reset_task_state()
        obs = env.reset()
        validate_obs_type(obs)
        for _step_index in range(3):
            action = env.action_space.sample()
            step_return = env.step(action)
            validate_step_return_type(step_return)


def validate_single_task_env(env: gym.Env) -> None:
    for _episode in range(10):
        obs = env.reset()
        valiate_obs_type_single_env(obs)
        for _ in range(3):
            action = env.action_space.sample()
            step_return = env.step(action)
            validate_step_return_type_single_env(step_return)
