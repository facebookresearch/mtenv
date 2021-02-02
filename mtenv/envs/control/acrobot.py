# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the  LICENSE file in the root directory of this source tree.


import numpy as np
from gym import spaces
from numpy import cos, pi, sin

from mtenv import MTEnv
from mtenv.utils import seeding

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = [
    "Alborz Geramifard",
    "Robert H. Klein",
    "Christoph Dann",
    "William Dabney",
    "Jonathan P. How",
]
__license__ = "BSD 3-Clause"
__author__ = "Christoph Dann <cdann@cdann.de>"

# SOURCE:
# https://github.com/rlpy/rlpy/blob/master/rlpy/Domains/Acrobot.py


class MTAcrobot(MTEnv):
    """A acrobot environment with varying characteristics
    The task descriptor is composed of values between -1 and +1 and mapped to acrobot physical characcteristics in the
    self._mu_to_vars function.


    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 15}

    dt = 0.2

    def _mu_to_vars(self, mu):
        self.LINK_LENGTH_1 = 1.0 + mu[0] * 0.5
        self.LINK_LENGTH_2 = 1.0 + mu[1] * 0.5
        self.LINK_MASS_1 = 1.0 + mu[2] * 0.5
        self.LINK_MASS_2 = 1.0 + mu[3] * 0.5
        self.LINK_COM_POS_1 = 0.5
        self.LINK_COM_POS_2 = 0.5
        if mu[6] > 0:
            self.AVAIL_TORQUE = [-1.0, 0.0, 1.0]
        else:
            self.AVAIL_TORQUE = [1.0, 0.0, -1.0]
        self.LINK_MOI = 1.0

    torque_noise_max = 0.0
    MAX_VEL_1 = 4 * pi + pi
    MAX_VEL_2 = 9 * pi + 2 * pi

    #: use dynamics equations from the nips paper or the book
    book_or_nips = "book"
    action_arrow = None
    domain_fig = None
    actions_num = 3

    def __init__(self):
        self.viewer = None
        self.action_space = spaces.Discrete(3)
        self.state = None
        high = np.array(
            [1.5, 1.5, 1.5, 1.5, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32
        )
        low = -high
        observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        action_space = spaces.Discrete(3)
        high = np.array([1.0 for k in range(5)])
        task_space = spaces.Box(-high, high, dtype=np.float32)
        super().__init__(
            action_space=action_space,
            env_observation_space=observation_space,
            task_observation_space=task_space,
        )

    def step(self, a):
        self.t += 1
        self._mu_to_vars(self.task_state)
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random_env.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = -1.0 if not terminal else 0.0
        return (
            {"env_obs": self._get_obs(), "task_obs": self.get_task_obs()},
            reward,
            terminal,
            {},
        )

    def reset(self):
        self._mu_to_vars(self.task_state)
        self.state = self.np_random_env.uniform(low=-0.1, high=0.1, size=(4,))
        self.t = 0
        return {"env_obs": self._get_obs(), "task_obs": self.get_task_obs()}

    def get_task_obs(self):
        return self.task_state

    def get_task_state(self):
        return self.task_state

    def set_task_state(self, task_state):
        self.task_state = task_state

    def _get_obs(self):
        s = self.state
        return [cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]]

    def _terminal(self):
        s = self.state
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

    def _dsdt(self, s_augmented, t):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = (
            m1 * lc1 ** 2
            + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2))
            + I1
            + I2
        )
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2
            ) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return (dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0)

    def seed(self, env_seed):
        self.np_random_env, seed = seeding.np_random(env_seed)
        return [seed]

    def seed_task(self, task_seed):
        self.np_random_task, seed = seeding.np_random(task_seed)
        return [seed]

    def sample_task_state(self):
        self.assert_task_seed_is_set()
        super().sample_task_state()
        new_task_state = [
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
            self.np_random_task.uniform(-1, 1),
        ]
        return new_task_state


def wrap(x, m, M):
    """
    :param x: a scalar
    :param m: minimum possible value in range
    :param M: maximum possible value in range
    Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """
    :param x: scalar
    Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):

        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout


class Acrobot(MTAcrobot):
    """The original acrobot environment in the MTEnv fashion"""

    def __init__(self):
        super().__init__()

    def sample_task_state(self):
        self.assert_task_seed_is_set()
        super().sample_task_state()
        new_task_state = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        return new_task_state


if __name__ == "__main__":
    env = MTAcrobot()
    env.seed(5)
    env.seed_task(15)
    env.reset_task_state()
    obs = env.reset()
    print(obs)
    done = False
    while not done:
        obs, rew, done, _ = env.step(np.random.randint(env.action_space.n))
        print(obs)
