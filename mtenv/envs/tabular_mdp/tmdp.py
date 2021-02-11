# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.special
from gym import spaces
from gym.utils import seeding

from mtenv import MTEnv


class TMDP(MTEnv):
    """Defines a Tabuular MDP where task_state is the reward matrix,transition matrix
        reward_matrix is n_states*n_actions and gies the probability of having a reward = +1 when choosing action a in state s (matrix[s,a])
        transition_matrix is n_states*n_actions*n_states and gives the probability of moving to state s' when choosing action a in state s (matrix[s,a,s'])
    Args:
        MTEnv ([type]): [description]
    """

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        ohigh = np.array([1.0 for n in range(n_states + 1)])
        olow = np.array([0.0 for n in range(n_states + 1)])
        observation_space = spaces.Box(olow, ohigh, dtype=np.float32)
        action_space = spaces.Discrete(n_actions)
        self.task_state = (
            np.zeros((n_states, n_actions)),
            np.zeros((n_states, n_actions, n_states)),
        )
        o = self.get_task_obs()
        thigh = np.ones((len(o),))
        tlow = np.zeros((len(o),))
        task_space = spaces.Box(tlow, thigh, dtype=np.float32)
        super().__init__(
            action_space=action_space,
            env_observation_space=observation_space,
            task_observation_space=task_space,
        )

        # task state is the reward matrix and transition matrix

    def get_task_obs(self):
        obs = list(self.task_state[0].flatten()) + list(self.task_state[1].flatten())
        return obs

    def get_task_state(self):
        return self.task_state

    def set_task_state(self, task_state):
        self.task_state = task_state

    def sample_task_state(self):
        raise NotImplementedError

    def seed(self, env_seed):
        self.np_random_env, seed = seeding.np_random(env_seed)
        return [seed]

    def seed_task(self, task_seed):
        self.np_random_task, seed = seeding.np_random(task_seed)
        return [seed]

    def step(self, action):
        t_reward, t_matrix = self.task_state
        reward = 0.0

        if self.np_random_env.rand() < t_reward[self.state][action]:
            reward = 1.0
        self.state = self.np_random_env.multinomial(
            1, t_matrix[self.state][action]
        ).argmax()

        obs = np.zeros(self.n_states + 1)
        obs[self.state] = 1.0
        obs[-1] = reward
        return (
            {"env_obs": list(obs), "task_obs": self.get_task_obs()},
            reward,
            False,
            {},
        )

    def reset(self):
        self.state = self.np_random_env.randint(self.n_states)
        obs = np.zeros(self.n_states + 1)
        obs[self.state] = 1.0
        return {"env_obs": list(obs), "task_obs": self.get_task_obs()}


class UniformTMDP(TMDP):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)

    def sample_task_state(self):
        self.assert_task_seed_is_set()
        t_reward = self.np_random_task.rand(self.n_states, self.n_actions)
        t_transitions = self.np_random_task.randn(
            self.n_states, self.n_actions, self.n_states
        )
        t_transitions = scipy.special.softmax(t_transitions, axis=2)

        new_task_state = t_reward, t_transitions
        return new_task_state


if __name__ == "__main__":
    env = UniformTMDP(3, 2)
    env.seed(5)
    env.seed_task(14)
    env.reset_task_state()
    obs = env.reset()
    done = False
    while not done:
        action = np.random.randint(env.action_space.n)
        obs, rew, done, _ = env.step(action)
        print(obs["env_obs"])
