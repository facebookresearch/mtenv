MTEnv
=====

Environment interface for multi-task reinforcement learning.

Introduction
------------

MTEnv is a library to interface with environments for multi-task reinforcement learning. It has two main components:


* A core API/interface that extends the `gym interface <https://gym.openai.com/>`_ and adds first-class support for multi-task RL.

* A `collection of environments <http://localhost:8000/pages/envs.html>`_ that implement the API.

Together, these two components should provide a standard interface for multi-task RL environments and make it easier to reuse components and tools across environments.

You can read more about the difference between ``MTEnv`` and single-task environments `here. <http://localhost:8000/pages/readme.html#multitask-observation>`_

List of publications & submissions using MTEnv (please create a pull request to add the missing entries):
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


* `Learning Robust State Abstractions for Hidden-Parameter Block MDPs <https://arxiv.org/abs/2007.07206>`_

License
^^^^^^^

MTEnv is licensed under `MIT License <LICENSE>`_.

Citing MTEnv
^^^^^^^^^^^^

If you use MTEnv in your research, please use the following BibTeX entry:

.. code-block::

   @Misc{Sodhani2020MTEnv,
     author =       {Shagun Sodhani, Ludovic Denoyer, Pierre-Alexandre Kamienny, Olivier Delalleau},
     title =        {MTEnv - Environment interface for mulit-task reinforcement learning},
     howpublished = {Github},
     year =         {2021},
     url =          {https://github.com/facebookresearch/mtenv}
   }

Installation
------------

MTEnv has two components - a core API and environments that implement the API.

The **Core API** can be installed via ``pip install mtenv``. 

The **list of environments**\ , that implement the API, is available `here <http://localhost:8000/pages/envs.html>`_. Any of these environments can be installed via ``pip install "mtenv[env_name]"``. For example, the ``MetaWorld`` environment can be installed via ``pip install "mtenv[metaworld]"``.

All the environments can be installed at once using ``pip install "mtenv[all]"``. However, note that some environments may have incompatible dependencies.

MTEnv can also be installed from the source by first cloning the repo (\ ``git clone git@github.com:fairinternal/mtenv.git``\ ), *cding* into the directory ``cd mtenv``\ , and then using the pip commands as described above. For example, ``pip install mtenv`` to install the core API, and ``pip install "mtenv[env_name]"`` to install a particular environment.

Usage
------------

MTEnv provides an interface very similar to the standard gym environments.
One key difference between multitask environments (that implement the MTEnv
interface and single tasks environments is in terms of observation that
they return.

.. _multitask_observation:

MultiTask Observation
^^^^^^^^^^^^^^^^^^^^^

The multitask environments returns a dictionary as the observation. This
dictionary has two keys: (i) `env_obs` which maps to the observation from
the environment (i.e. the observation that a single task environments return)
and (ii) `task_obs` which maps to the task-specific information from the
environment. In the simplest case, `task_obs` can be an integer denoting
the task index. In other cases, `task_obs` can provide richer information.

.. code-block:: python

    from mtenv import make
    env = make("MT-MetaWorld-MT10-v0")
    obs = env.reset()
    print(obs)
    # {'env_obs': array([-0.03265039,  0.51487777,  0.2368754 , -0.06968209,  0.6235982 ,
    #    0.01492813,  0.        ,  0.        ,  0.        ,  0.03933976,
    #    0.89743189,  0.01492813]), 'task_obs': 1}
    action = env.action_space.sample()
    print(action)
    # array([-0.76422   , -0.15384133,  0.74575615, -0.11724994], dtype=float32)
    obs, reward, done, info = env.step(action)
    print(obs)
    # {'env_obs': array([-0.02583682,  0.54065546,  0.22773503, -0.06968209,  0.6235982 ,
    #    0.01494118,  0.        ,  0.        ,  0.        ,  0.03933976,
    #    0.89743189,  0.01492813]), 'task_obs': 1}



Documentation
-------------

`https://mtenv.readthedocs.io <https://mtenv.readthedocs.io>`_

Contributing to MTEnv
---------------------

There are several ways to contribute to MTEnv.


#. 
   Contribute a new environment. We currently support `three environment suites <http://localhost:8000/pages/envs.html>`_ via MTEnv and are looking forward to adding more environments. Contributors will be added as authors of the library. You can learn more about the workflow of adding an environment `here. <https://github.com>`_

#. 
   Use MTEnv in your research.

#. 
   Check out the `beginner-friendly <https://github.com>`_ issues on GitHub and contribute to fixing those issues.

Local Setup
^^^^^^^^^^^

Follow these instructions to setup MTEnv locally:


* Clone locally - ``git clone git@github.com:fairinternal/mtenv.git``.
* *cd* into the directory - ``cd mtenv``.
* Install MTEnv in the dev mode - ``pip install -e ".[dev]"``
* Install pre-commit hooks - ``pre-commit install``
* Tests can be run locally using ``nox``. The code is linted using:

  * ``black``
  * ``flake8``
  * ``mypy``

Community
---------

Ask questions in the chat or github issues:


* `Chat <https://mtenv.zulipchat.com>`_
* `Issues <https://https://github.com/facebookresearch/mtenv/issues>`_

Glossary
--------

.. _task_state:

Task State
^^^^^^^^^^

Task State contains all the information that the environment needs to
switch to any other task.