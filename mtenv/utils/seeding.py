# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple

from gym.utils import seeding
from numpy.random import RandomState


def np_random(seed: Optional[int]) -> Tuple[RandomState, int]:
    rng, seed = seeding.np_random(seed)
    assert isinstance(seed, int)
    return rng, seed
