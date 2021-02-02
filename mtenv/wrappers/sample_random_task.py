# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from mtenv.core import MTEnv
from mtenv.utils.types import ObsType
from mtenv.wrappers.multitask import MultiTask as MultiTaskWrapper


class SampleRandomTask(MultiTaskWrapper):
    """
    A wrapper that samples a new task at each env.reset() call
    """

    def __init__(self, env: MTEnv):
        super().__init__(env=env)

    def reset(self) -> ObsType:
        self.env.reset_task_state()
        return self.env.reset()
