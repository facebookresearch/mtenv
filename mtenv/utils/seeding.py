# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Optional, Tuple

from gym.utils import seeding


def np_random(seed: Optional[int]) -> Tuple[seeding.RandomNumberGenerator, int]:
    """Set the seed for numpy's random generator.

    Args:
        seed (Optional[int]):

    Returns:
        Tuple[RandomNumberGenerator, int]: Returns a tuple of random
            number generator and seed.
    """
    rng, seed = seeding.np_random(seed)
    assert isinstance(seed, int)
    return rng, seed
