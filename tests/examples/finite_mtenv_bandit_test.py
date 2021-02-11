# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import pytest

from examples.finite_mtenv_bandit import FiniteMTBanditEnv  # noqa: E402
from tests.utils.utils import validate_mtenv


def get_valid_n_tasks_and_arms() -> List[int]:
    return [(1, 2), (10, 20), (100, 200)]


def get_invalid_n_tasks_and_arms() -> List[int]:
    return [(-1, 2), (0, 3), (1, -2), (3, 0)]


@pytest.mark.parametrize("n_tasks, n_arms", get_valid_n_tasks_and_arms())
def test_mtenv_bandit_with_valid_input(n_tasks, n_arms):
    env = FiniteMTBanditEnv(n_tasks=n_tasks, n_arms=n_arms)
    validate_mtenv(env=env)


@pytest.mark.parametrize("n_tasks, n_arms", get_invalid_n_tasks_and_arms())
def test_mtenv_bandit_with_invalid_input(n_tasks, n_arms):
    with pytest.raises(Exception):
        env = FiniteMTBanditEnv(n_tasks=n_tasks, n_arms=n_arms)
        validate_mtenv(env=env)
