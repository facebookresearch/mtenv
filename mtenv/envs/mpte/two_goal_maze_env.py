# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the  LICENSE file in the root directory of this source tree.

import copy
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gym import spaces
from gym.spaces.box import Box as BoxSpace
from gym.spaces.dict import Dict as DictSpace
from gym.spaces.discrete import Discrete as DiscreteSpace
from gym_miniworld.entity import Box
from gym_miniworld.miniworld import Agent, MiniWorldEnv
from numpy.random import RandomState

from mtenv.utils import seeding
from mtenv.utils.types import DoneType, InfoType, RewardType, TaskObsType
from mtenv.wrappers.env_to_mtenv import EnvToMTEnv

TaskStateType = List[int]

ActionType = int

EnvObsType = Dict[str, Union[int, List[int], List[float]]]
ObsType = Dict[str, Union[EnvObsType, TaskObsType]]
StepReturnType = Tuple[ObsType, RewardType, DoneType, InfoType]


class MTMiniWorldEnv(EnvToMTEnv):
    def make_observation(self, env_obs: EnvObsType) -> ObsType:
        raise NotImplementedError

    def get_task_obs(self) -> TaskObsType:
        return self.env.get_task_obs()

    def get_task_state(self) -> TaskStateType:
        return self.env.task_state

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.env.set_task_state(task_state)

    def sample_task_state(self) -> TaskStateType:
        return self.env.sample_task_state()

    def reset(self, **kwargs: Dict[str, Any]) -> ObsType:  # type: ignore[override]
        # signature is incompatible with supertype.
        self.assert_env_seed_is_set()
        return self.env.reset(**kwargs)

    def step(self, action: ActionType) -> StepReturnType:  # type: ignore
        return self.env.step(action)

    def assert_env_seed_is_set(self) -> None:
        assert self.env.np_random_env is not None, "please call `seed()` first"

    def assert_task_seed_is_set(self) -> None:
        assert self.env.np_random_task is not None, "please call `seed_task()` first"

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for environment observations"""
        return self.env.seed(seed=seed)


class TwoGoalMazeEnv(MiniWorldEnv):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        size_x=5,
        size_y=5,
        obs_type="xy",
        task_seed=0,
        n_tasks=10,
        p_change=0.0,
        empty_mu=False,
    ):
        assert p_change == 0.0
        self.empty_mu = empty_mu
        self.obs_type = obs_type
        self.seed_task(seed=task_seed)
        self.np_random_env: Optional[RandomState] = None
        self.size_x, self.size_y = size_x, size_y
        self.task_state = []

        super().__init__()
        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)
        if self.obs_type == "xy":
            _obs_space = BoxSpace(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )

        else:
            _obs_space = BoxSpace(
                low=-1.0,
                high=1.0,
                shape=(64, 64),
                dtype=np.float32,
            )
        self.observation_space = DictSpace(
            {
                "obs": _obs_space,
                "total_reward": BoxSpace(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
            }
        )

    def assert_env_seed_is_set(self) -> None:
        """Check that the env seed is set."""
        assert self.np_random_env is not None, "please call `seed()` first"

    def assert_task_seed_is_set(self) -> None:
        """Check that the task seed is set."""
        assert self.np_random_task is not None, "please call `seed_task()` first"

    def seed_task(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for task information"""
        self.np_random_task, seed = seeding.np_random(seed)
        assert isinstance(seed, int)
        return [seed]

    def sample_task_state(self) -> TaskStateType:
        self.assert_task_seed_is_set()
        return [self.np_random_task.randint(2)]

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.task_state = task_state

    def _gen_world(self):
        self.reset_task_state()
        room1 = self.add_rect_room(
            min_x=-self.size_x,
            max_x=self.size_x,
            min_z=-self.size_y,
            max_z=self.size_y,
            wall_tex="brick_wall",
        )
        self.room1 = room1
        room2 = self.add_rect_room(
            min_x=-self.size_x,
            max_x=self.size_x,
            min_z=self.size_y,
            max_z=self.size_y + 1,
            wall_tex="cardboard",
        )
        self.connect_rooms(room1, room2, min_x=-self.size_x, max_x=self.size_x)

        room3 = self.add_rect_room(
            min_x=-self.size_x,
            max_x=self.size_x,
            min_z=-self.size_y - 1,
            max_z=-self.size_y,
            wall_tex="lava",
        )
        self.connect_rooms(room1, room3, min_x=-self.size_x, max_x=self.size_x)

        room4 = None
        if self.task_state[0] == 0:
            room4 = self.add_rect_room(
                min_x=-self.size_x - 1,
                max_x=-self.size_x,
                min_z=-self.size_y,
                max_z=self.size_y,
                wall_tex="wood_planks",
            )
        else:
            room4 = self.add_rect_room(
                min_x=-self.size_x - 1,
                max_x=-self.size_x,
                min_z=-self.size_y,
                max_z=self.size_y,
                wall_tex="slime",
            )

        self.connect_rooms(room1, room4, min_z=-self.size_y, max_z=self.size_y)

        room5 = self.add_rect_room(
            min_x=self.size_x,
            max_x=self.size_x + 1,
            min_z=-self.size_y,
            max_z=self.size_y,
            wall_tex="metal_grill",
        )

        self.connect_rooms(room1, room5, min_z=-self.size_y, max_z=self.size_y)

        self.boxes = []
        self.boxes.append(Box(color="blue"))
        self.boxes.append(Box(color="red"))
        self.place_entity(self.boxes[0], room=room1)
        self.place_entity(self.boxes[1], room=room1)

        # Choose a random room and position to spawn at
        _dir = self.np_random_env.randint(8) * (math.pi / 4) - math.pi
        self.place_agent(
            dir=_dir,
            room=room1,
        )
        while self._dist() < 2 or self._ndist() < 2:
            self.place_agent(
                dir=_dir,
                room=room1,
            )

    def _dist(self):
        bp = self.boxes[int(self.task_state[0])].pos
        pos = self.agent.pos
        distance = math.sqrt((bp[0] - pos[0]) ** 2 + (bp[2] - pos[2]) ** 2)

        return distance

    def _ndist(self):
        bp = self.boxes[1 - int(self.task_state[0])].pos
        pos = self.agent.pos
        distance = math.sqrt((bp[0] - pos[0]) ** 2 + (bp[2] - pos[2]) ** 2)

        return distance

    def reset(self) -> ObsType:
        self.assert_env_seed_is_set()
        self.max_episode_steps = 200
        self.treward = 0.0
        self.step_count = 0
        self.agent = Agent()
        self.entities: List[Any] = []
        self.rooms: List[Any] = []
        self.wall_segs: List[Any] = []
        self._gen_world()
        self.blocked = False
        rand = self.rand if self.domain_rand else None
        self.params.sample_many(
            rand, self, ["sky_color", "light_pos", "light_color", "light_ambient"]
        )

        for ent in self.entities:
            ent.randomize(self.params, rand)

        # Compute the min and max x, z extents of the whole floorplan
        self.min_x = min(r.min_x for r in self.rooms)
        self.max_x = max(r.max_x for r in self.rooms)
        self.min_z = min(r.min_z for r in self.rooms)
        self.max_z = max(r.max_z for r in self.rooms)

        # Generate static data
        if len(self.wall_segs) == 0:
            self._gen_static_data()

        # Pre-compile static parts of the environment into a display list
        self._render_static()
        _pos = [
            (self.agent.pos[0] / self.size_x) * 2.1 - 1.0,
            (self.agent.pos[2] / self.size_y) * 2.1 - 1.0,
        ]
        _dir = [self.agent.dir_vec[0], self.agent.dir_vec[2]]

        if self.obs_type == "xy":
            _mu = [0.0]
            at = math.atan2(_dir[0], _dir[1])
            o = copy.deepcopy(_pos + [at] + _mu)
        else:
            o = (self.render_obs() / 255.0) * 2.0 - 1.0

        return self.make_obs(env_obs=o, total_reward=[0.0])

    def get_task_obs(self) -> TaskObsType:
        mmu = copy.deepcopy(self.task_state)
        if self.empty_mu:
            mmu = [0.0]
        return mmu

    def get_task_state(self) -> TaskStateType:
        return self.task_state

    def reset_task_state(self) -> None:
        """Sample a new task_state and set that as the new task_state"""
        self.set_task_state(task_state=self.sample_task_state())

    def make_obs(self, env_obs: Any, total_reward: List[float]) -> ObsType:

        return {
            "env_obs": {"obs": env_obs, "total_reward": total_reward},
            "task_obs": self.get_task_obs(),
        }

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Set the seed for environment observations"""
        self.np_random_env, seed = seeding.np_random(seed)
        return [seed] + super().seed(seed=seed)

    def step(self, action: ActionType) -> StepReturnType:
        self.step_count += 1
        if not self.blocked:
            if action == 2:
                self.move_agent(0.51, 0.0)  # fwd_step, fwd_drift)
            elif action == 0:
                self.turn_agent(45)
            elif action == 1:
                self.turn_agent(-45)
        reward = 0.0
        done = False

        distance = self._dist()
        if distance < 2:
            reward = +1.0
            done = True
        distance = self._ndist()
        if distance < 2:
            reward = -1.0
            done = True
        _pos = [
            (self.agent.pos[0] / self.size_x) * 2.1 - 1.0,
            (self.agent.pos[2] / self.size_y) * 2.1 - 1.0,
        ]
        _dir = [self.agent.dir_vec[0], self.agent.dir_vec[2]]

        if self.obs_type == "xy":
            at = math.atan2(_dir[0], _dir[1])
            _mu = [0.0]
            if (at < -1.5 and at > -1.7) and not self.empty_mu:
                _mu = [1.0]
                if self.task_state[0] == 0:
                    _mu = [-1.0]

            o = copy.deepcopy(_pos + [at] + _mu)
        else:
            o = (self.render_obs() / 255.0) * 2.0 - 1.0

        self.treward += reward

        return self.make_obs(env_obs=o, total_reward=[self.treward]), reward, done, {}


def build_two_goal_maze_env(size_x: int, size_y: int, task_seed: int, n_tasks: int):
    env = MTMiniWorldEnv(
        TwoGoalMazeEnv(
            size_x=size_x, size_y=size_y, task_seed=task_seed, n_tasks=n_tasks
        ),
        task_observation_space=DiscreteSpace(n=1),
    )
    return env
