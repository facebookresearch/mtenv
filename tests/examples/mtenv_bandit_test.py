# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

import pytest

from examples.mtenv_bandit import MTBanditEnv  # noqa: E402
from tests.utils.utils import validate_mtenv


def get_valid_n_arms() -> List[int]:
    return [1, 10, 100]


def get_invalid_n_arms() -> List[int]:
    return [-1, 0]


@pytest.mark.parametrize("n_arms", get_valid_n_arms())
def test_ntasks_id_wrapper_with_valid_input(n_arms):
    env = MTBanditEnv(n_arms=n_arms)
    validate_mtenv(env=env)


@pytest.mark.parametrize("n_arms", get_invalid_n_arms())
def test_ntasks_id_wrapper_with_invalid_input(n_arms):
    with pytest.raises(Exception):
        env = MTBanditEnv(n_arms=n_arms)
        validate_mtenv(env=env)
