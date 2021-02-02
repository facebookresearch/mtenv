# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper to fix the number of tasks in an existing multitask environment
and return the id of the task as part of the observation."""

from gym.spaces import Dict as DictSpace
from gym.spaces import Discrete

from mtenv import MTEnv
from mtenv.utils.types import ActionType, ObsType, StepReturnType, TaskStateType
from mtenv.wrappers.ntasks import NTasks


class NTasksId(NTasks):
    def __init__(self, env: MTEnv, n_tasks: int):
        """Wrapper to fix the number of tasks in an existing multitask
        environment to `n_tasks`.

        Each task is sampled in this fixed set of `n_tasks`. The agent
        observes the id of the task.

        Args:
            env (MTEnv): Multitask environment to wrap over.
            n_tasks (int): Number of tasks to sample.
        """
        self.env = env

        super().__init__(n_tasks=n_tasks, env=env)
        self.task_state: TaskStateType
        self.observation_space: DictSpace = DictSpace(
            spaces={
                "env_obs": self.observation_space["env_obs"],
                "task_obs": Discrete(n_tasks),
            }
        )

    def _update_obs(self, obs: ObsType) -> ObsType:
        obs["task_obs"] = self.get_task_obs()
        return obs

    def step(self, action: ActionType) -> StepReturnType:
        obs, reward, done, info = self.env.step(action)
        return self._update_obs(obs), reward, done, info

    def get_task_obs(self) -> TaskStateType:
        return self.task_state

    def get_task_state(self) -> TaskStateType:
        return self.task_state

    def set_task_state(self, task_state: TaskStateType) -> None:
        self.env.set_task_state(self.tasks[task_state])
        self.task_state = task_state

    def reset(self) -> ObsType:
        obs = self.env.reset()
        return self._update_obs(obs)

    def sample_task_state(self) -> TaskStateType:
        self.assert_task_seed_is_set()
        if not self._are_tasks_set:
            self.tasks = [self.env.sample_task_state() for _ in range(self.n_tasks)]
            self._are_tasks_set = True

        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        id_task = self.np_random_task.randint(self.n_tasks)  # type: ignore[union-attr]
        return id_task
