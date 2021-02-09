
How to create new environments
---------------------------

There are two workflows:


#. 
   You have a standard gym environment, which you want to convert into a multitask environment. For example, ``examples/bandit.py`` implements ``BanditEnv`` which is a standard multi-arm bandit, without an explicit notion of task. The user has the following options:


   * 
     Write a new subclass, say ``MTBanditEnv`` (which subclasses ``MTEnv``\ ) as shown in ``examples/mtenv_bandit.py``.

   * 
     Use the ``EnvToMTEnv`` wrapper and wrap the existing single task environment. In some cases, the wrapper may have to be extended, as is done in ``examples/wrapped_bandit.py``.

#. 
   If you do not have a single-task gym environment to start with, it is recommended that you directly extend the ``MTEnv`` class. Implementations in ``mtenv/envs`` can be seen as a reference.

If you want to contribute an environment to the repo, checkout the `Contribution Guide <https://github.com/facebookresearch/mtenv/blob/main/.github/CONTRIBUTING.md>`.

