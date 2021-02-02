# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import pytest
from gym import spaces

from examples.bandit import BanditEnv  # noqa: E402
from examples.wrapped_bandit import MTBanditWrapper  # noqa: E402
from tests.utils.utils import validate_mtenv


def get_valid_n_arms() -> List[int]:
    return [1, 10, 100]


def get_invalid_n_arms() -> List[int]:
    return [-1, 0]


@pytest.mark.parametrize("n_arms", get_valid_n_arms())
def test_ntasks_id_wrapper_with_valid_input(n_arms):

    env = MTBanditWrapper(
        env=BanditEnv(n_arms),
        task_observation_space=spaces.Box(low=0.0, high=1.0, shape=(n_arms,)),
    )

    validate_mtenv(env=env)


@pytest.mark.parametrize("n_arms", get_invalid_n_arms())
def test_ntasks_id_wrapper_with_invalid_input(n_arms):
    with pytest.raises(Exception):
        env = MTBanditWrapper(
            env=BanditEnv(n_arms),
            task_observation_space=spaces.Box(low=0.0, high=1.0, shape=(n_arms,)),
        )
        validate_mtenv(env=env)
