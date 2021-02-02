
Supported Environments
======================

The following environments are supported:

Control
-------

**Installation**

.. code-block:: bash

    pip install "mtenv[control]"

HiPBMDP
-------

:cite:`mtrl_as_a_hidden_block_mdp` create a family of MDPs using the
existing environment-task pairs from DeepMind Control Suite :cite:`tassa2020dmcontrol`
and change one environment parameter to sample different MDPs. For more details,
refer :cite:`mtrl_as_a_hidden_block_mdp`.


**Installation**

.. code-block:: bash

    pip install "mtenv[hipbmdp]"

**Usage**

.. code-block:: python

    from mtenv import make
    env = make("MT-HiPBMDP-Finger-Spin-vary-size-v0")
    env.reset()


MetaWorld
---------

:cite:`yu2020meta` proposed an open-source simulated benchmark for
meta-reinforcement learning and multi-task learning consisting of 50 distinct
robotic manipulation tasks. For more details, refer :cite:`yu2020meta`.
MTEnv provides a wrapper for the multi-task learning environments.

**Installation**

.. code-block:: bash

    pip install "mtenv[metaworld]"

**Usage**

.. code-block:: python

    from mtenv import make
    env = make("MT-MetaWorld-MT10-v0") # or MT-MetaWorld-MT50-v0 or MT-MetaWorld-MT1-v0
    env.reset()

MPTE
----

**Installation**

.. code-block:: bash

    pip install "mtenv[mpte]"


References
-------------

.. bibliography::