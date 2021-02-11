# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List

import pytest

from examples.bandit import BanditEnv  # noqa: E402
from tests.utils.utils import validate_single_task_env


def get_valid_n_arms() -> List[int]:
    return [1, 10, 100]


def get_invalid_n_arms() -> List[int]:
    return [-1, 0]


@pytest.mark.parametrize("n_arms", get_valid_n_arms())
def test_n_arm_bandit_with_valid_input(n_arms):
    env = BanditEnv(n_arms=n_arms)
    env.seed(seed=5)
    validate_single_task_env(env)


@pytest.mark.parametrize("n_arms", get_invalid_n_arms())
def test_n_arm_bandit_with_invalid_input(n_arms):
    with pytest.raises(Exception):
        env = BanditEnv(n_arms=n_arms)
        env.seed(seed=5)
        validate_single_task_env(env)
