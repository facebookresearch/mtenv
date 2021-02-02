# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper that samples a new task everytime the environment is reset."""

from mtenv import MTEnv
from mtenv.utils.types import ObsType
from mtenv.wrappers.multitask import MultiTask


class SampleRandomTask(MultiTask):
    def __init__(self, env: MTEnv):
        """Wrapper that samples a new task everytime the environment is
        reset.

        Args:
            env (MTEnv): Multitask environment to wrap over.
        """

        super().__init__(env=env)

    def reset(self) -> ObsType:
        self.env.reset_task_state()
        return self.env.reset()
