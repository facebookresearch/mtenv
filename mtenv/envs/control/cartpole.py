# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import math

import numpy as np
from gym import logger, spaces

from mtenv import MTEnv
from mtenv.utils import seeding

"""
Classic cart-pole system implemented based on Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""


class MTCartPole(MTEnv):
    """A cartpole environment with varying physical values
    (see the self._mu_to_vars function)
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def _mu_to_vars(self, mu):
        self.gravity = 9.8 + mu[0] * 5
        self.masscart = 1.0 + mu[1] * 0.5
        self.masspole = 0.1 + mu[2] * 0.09
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5 + mu[3] * 0.3
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10 * mu[4]
        if mu[4] == 0:
            self.force_mag = 10

    def __init__(self):
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360

        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ]
        )
        observation_space = spaces.Box(-high, high, dtype=np.float32)
        action_space = spaces.Discrete(2)
        high = np.array([1.0 for k in range(5)])
        task_space = spaces.Box(-high, high, dtype=np.float32)
        super().__init__(
            action_space=action_space,
            env_observation_space=observation_space,
            task_observation_space=task_space,
        )

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        # Angle at which to fail the episode

        self.state = None
        self.steps_beyond_done = None

        self.task_state = None

    def step(self, action):
        self.t += 1
        self._mu_to_vars(self.task_state)

        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = [x, x_dot, theta, theta_dot]
        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        reward = 0

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )
                print(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return (
            {"env_obs": self.state, "task_obs": self.get_task_obs()},
            reward,
            done,
            {},
        )

    def reset(self, **args):
        self.assert_env_seed_is_set()
        assert self.task_state is not None

        self._mu_to_vars(self.task_state)
        self.state = self.np_random_env.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.t = 0
        return {"env_obs": self.state, "task_obs": self.get_task_obs()}

    def get_task_obs(self):
        return self.task_state

    def get_task_state(self):
        return self.task_state

    def set_task_state(self, task_state):
        self.task_state = task_state

    def sample_task_state(self):
        self.assert_task_seed_is_set()
        super().sample_task_state()
        new_task_state = [
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
        ]
        return new_task_state

    def seed(self, env_seed):
        self.np_random_env, seed = seeding.np_random(env_seed)
        return [seed]

    def seed_task(self, task_seed):
        self.np_random_task, seed = seeding.np_random(task_seed)
        return [seed]


class CartPole(MTCartPole):
    """The original cartpole environment in the MTEnv fashion"""

    def __init__(self):
        super().__init__()

    def sample_task_state(self):
        new_task_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        return new_task_state


if __name__ == "__main__":
    env = MTCartPole()
    env.seed(5)
    env.seed_task(15)
    env.reset_task_state()
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        obs, rew, done, _ = env.step(np.random.randint(env.action_space.n))
        print(obs)
