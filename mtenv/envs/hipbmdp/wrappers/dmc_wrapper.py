# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Any, Dict, Optional

import dmc2gym
import numpy as np
from dmc2gym.wrappers import DMCWrapper as BaseDMCWrapper
from gym import spaces

import local_dm_control_suite as local_dmc_suite


class DMCWrapper(BaseDMCWrapper):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        task_kwargs: Any = None,
        visualize_reward: Optional[Dict[str, Any]] = None,
        from_pixels: bool = False,
        height=84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        environment_kwargs: Any = None,
        channels_first: bool = True,
    ):
        """This wrapper is based on implementation from
        https://github.com/denisyarats/dmc2gym/blob/master/dmc2gym/wrappers.py#L37

        We extend the wrapper so that we can use the modified version of
        `dm_control_suite`.
        """
        assert (
            "random" in task_kwargs  # type: ignore [operator]
        ), "please specify a seed, for deterministic behaviour"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first
        if visualize_reward is None:
            visualize_reward = {}
        # create task
        self._env = local_dmc_suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # true and normalized action spaces
        self._true_action_space = dmc2gym.wrappers._spec_to_box(
            [self._env.action_spec()]
        )
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = dmc2gym.wrappers._spec_to_box(
                self._env.observation_spec().values()
            )

        self._state_space = dmc2gym.wrappers._spec_to_box(
            self._env.observation_spec().values()
        )

        self.current_state = None

        # set seed
        self.seed(seed=task_kwargs["random"])  # type: ignore [index]
