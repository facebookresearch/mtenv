# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""Wrapper to fix the number of tasks in an existing multitask environment."""

from typing import List

from mtenv import MTEnv
from mtenv.utils.types import TaskStateType
from mtenv.wrappers.multitask import MultiTask


class NTasks(MultiTask):
    def __init__(self, env: MTEnv, n_tasks: int):
        """Wrapper to fix the number of tasks in an existing multitask
        environment to `n_tasks`.

        Each task is sampled in this fixed set of `n_tasks`.

        Args:
            env (MTEnv): Multitask environment to wrap over.
            n_tasks (int): Number of tasks to sample.
        """
        super().__init__(env=env)
        self.n_tasks = n_tasks
        self.tasks: List[TaskStateType]
        self._are_tasks_set = False

    def sample_task_state(self) -> TaskStateType:
        """Sample a `task_state` from the set of `n_tasks` tasks.

        `task_state` contains all the information that the environment
        needs to switch to any other task.

        The subclasses, extending this class, should ensure that the task
        seed is set (by calling `seed(int)`) before invoking this
        method (for reproducibility). It can be done by invoking
        `self.assert_task_seed_is_set()`.

        Returns:
            TaskStateType: For more information on `task_state`,
            refer :ref:`task_state`.
        """
        self.assert_task_seed_is_set()
        if not self._are_tasks_set:
            self.tasks = [self.env.sample_task_state() for _ in range(self.n_tasks)]
            self._are_tasks_set = True

        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        id_task = self.np_random_task.randint(self.n_tasks)  # type: ignore[union-attr]
        return self.tasks[id_task]

    def reset_task_state(self) -> None:
        """Sample a new task_state from the set of `n_tasks` tasks and
        set the environment to that `task_state`.

        For more information on `task_state`, refer :ref:`task_state`.
        """
        self.set_task_state(task_state=self.sample_task_state())
