MTEnv
=====

Environment interface for multi-task reinforcement learning

Installation
------------


* ``pip install ...``

**Install from source**


* ``git clone git@github.com:fairinternal/mtenv.git``
* ``cd mtenv``
* ``pip install mtenv``

Alternatively, ``pip install "git+https://git@github.com/fairinternal/mtenv.git@master#egg=mtenv"``

Documentation
-------------

TBD

Dev Setup
---------


* Clone locally and run ``pip install -e ".[dev]"``
* Install pre-commit hooks ``pre-commit install``
* The code is linted using:

  * ``black``
  * ``flake8``
  * ``mypy``

* Tests can be run locally using ``nox``

How to add new environments
---------------------------

There are two workflows:


* 
  The user have a standard gym environment, which they want to convert into a multitask environment. E.g.: ``examples/bandit.py`` has a ``BanditEnv`` which is a standard multi-arm bandit, without any explicit notion of task. The user has the following options:


  * 
    Write a new subclass, say MTBanditEnv (which subclasses MTEnv) as shown in ``examples/mtenv_bandit.py``.

  * 
    Use the ``MTEnvWrapper`` and wrap the existing standard model class. An example is shown in ``examples/wrapped_bandit.py``. 

* 
  If the user does not have a standard gym environment to start with, it is recommended that they directly extend the MTEnv class.

Running examples
^^^^^^^^^^^^^^^^


* Install from pypi or github.
* In the root folder, run ``PYTHONPATH=. python examples/<filename>.py``.

Pending items
-------------


#. `Google Doc <https://docs.google.com/document/d/1H98fJ-gI53kF1x99pt-7Gy_HPAE6q9DeLPeT3kBIThQ/edit>`_ to track some ongoing work.


.. _multitask_observation:

MultiTask Observation
=====================

One key difference between multitask environments (that implement the 
``mtenv`` interface and single tasks environments is in terms of ``observation``
that they return. The multitask environments returns a dictionary as the
observation. This dictionary has two keys: (i) ``env_obs`` which maps to
the observation from the environment (ie the observation that single task
environments return) and ``task_obs``\ which maps to the task-specific
information from the environment. In simple cases, ``task_obs`` can just be
the task index.

.. _task_state:

Task State
=====================

Task State contains all the information that the environment needs to
switch to any other task.