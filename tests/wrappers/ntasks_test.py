# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


from typing import List

import pytest

from mtenv.envs.control.cartpole import MTCartPole
from mtenv.wrappers.ntasks import NTasks as NTasksWrapper
from tests.utils.utils import validate_mtenv


def get_valid_num_tasks() -> List[int]:
    return [1, 10, 100]


def get_invalid_num_tasks() -> List[int]:
    return [-1, 0]


@pytest.mark.parametrize("n_tasks", get_valid_num_tasks())
def test_ntasks_wrapper_with_valid_input(n_tasks):
    env = MTCartPole()
    env = NTasksWrapper(env, n_tasks=n_tasks)
    validate_mtenv(env=env)


@pytest.mark.parametrize("n_tasks", get_invalid_num_tasks())
def test_ntasks_wrapper_with_invalid_input(n_tasks):
    with pytest.raises(Exception):
        env = MTCartPole()
        env = NTasksWrapper(env, n_tasks=n_tasks)
        validate_mtenv(env=env)
