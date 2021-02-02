# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List

from mtenv.core import MTEnv
from mtenv.utils.types import TaskStateType
from mtenv.wrappers.multitask import MultiTask


class NTasks(MultiTask):
    """
    A wrapper to fix the number of different tasks to n_tasks. Each task is sampled in the fixed set of n_tasks.
    """

    def __init__(self, env: MTEnv, n_tasks: int):
        super().__init__(env=env)
        self.n_tasks = n_tasks
        self.tasks: List[TaskStateType]
        self._are_tasks_set = False

    def sample_task_state(self) -> TaskStateType:
        self.assert_task_seed_is_set()
        if not self._are_tasks_set:
            self.tasks = [self.env.sample_task_state() for _ in range(self.n_tasks)]
            self._are_tasks_set = True

        # The assert statement (at the start of the function) ensures that self.np_random_task
        # is not None. Mypy is raising the warning incorrectly.
        id_task = self.np_random_task.randint(self.n_tasks)  # type: ignore[union-attr]
        return self.tasks[id_task]

    def reset_task_state(self) -> None:
        """Sample a new task_state and set that as the new task_state"""
        self.set_task_state(task_state=self.sample_task_state())
