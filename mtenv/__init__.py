# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
__version__ = "1.0"

from mtenv.core import MTEnv  # noqa: F401
from mtenv.envs.registration import make  # noqa: F401

__all__ = ["MTEnv", "make"]
