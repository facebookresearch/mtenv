[![CircleCI](https://circleci.com/gh/facebookresearch/mtenv.svg?style=svg&circle-token=d507c3b95e80c67d6d4daf2d43785df654af36d1)](https://circleci.com/gh/facebookresearch/mtenv)
![PyPI - License](https://img.shields.io/pypi/l/mtenv)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mtenv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)


# MTEnv
MultiTask Environments for Reinforcement Learning.

## Contents

1. [Introduction](#Introduction)

2. [Installation](#Installation)

3. [Usage](#Usage)

4. [Documentation](#Documentation)

5. [Contributing to MTEnv](#Contributing-to-MTEnv)

6. [Community](#Community)

## Introduction

MTEnv is a library to interface with environments for multi-task reinforcement learning. It has two main components:

* A core API/interface that extends the [gym interface](https://gym.openai.com/) and adds first-class support for multi-task RL.

* A [collection of environments](https://facebookresearch.github.io/mtenv/pages/envs.html) that implement the API.

Together, these two components should provide a standard interface for multi-task RL environments and make it easier to reuse components and tools across environments.

You can read more about the difference between `MTEnv` and single-task environments [here.](https://facebookresearch.github.io/mtenv/pages/readme.html#multitask-observation)

### List of publications & submissions using MTEnv (please create a pull request to add the missing entries):

* [Learning Adaptive Exploration Strategies in Dynamic Environments Through Informed Policy Regularization](https://arxiv.org/abs/2005.02934)

* [Learning Robust State Abstractions for Hidden-Parameter Block MDPs](https://arxiv.org/abs/2007.07206)

### License

* MTEnv uses [MIT License](https://github.com/facebookresearch/mtenv/blob/main/LICENSE).

* [Terms of Use](https://opensource.facebook.com/legal/terms)

* [Privacy Policy](https://opensource.facebook.com/legal/privacy)

### Citing MTEnv

If you use MTEnv in your research, please use the following BibTeX entry:
```
@Misc{Sodhani2020MTEnv,
  author =       {Shagun Sodhani, Ludovic Denoyer, Pierre-Alexandre Kamienny, Olivier Delalleau},
  title =        {MTEnv - Environment interface for mulit-task reinforcement learning},
  howpublished = {Github},
  year =         {2021},
  url =          {https://github.com/facebookresearch/mtenv}
}
```

## Installation

MTEnv has two components - a core API and environments that implement the API.

The **Core API** can be installed via `pip install mtenv`. 

The **list of environments**, that implement the API, is available [here](http://localhost:8000/pages/envs.html). Any of these environments can be installed via `pip install "mtenv[env_name]"`. For example, the `MetaWorld` environment can be installed via `pip install "mtenv[metaworld]"`.

All the environments can be installed at once using `pip install "mtenv[all]"`. However, note that some environments may have incompatible dependencies.

MTEnv can also be installed from the source by first cloning the repo (`git clone git@github.com:facebookresearch/mtenv.git`), *cding* into the directory `cd mtenv`, and then using the pip commands as described above. For example, `pip install mtenv` to install the core API, and `pip install "mtenv[env_name]"` to install a particular environment.

## Usage

MTEnv provides an interface very similar to the standard gym environments. One key difference between multi-task environments (that implement the MTEnv interface) and single-task environments is in terms of observation that they return.

### MultiTask Observation

The multi-task environments return a dictionary as the observation. This dictionary has two keys: (i) `env_obs`, which maps to the observation from the environment (i.e., the observation that a single task environments return), and (ii) `task_obs`, which maps to the task-specific information from the environment. In the simplest case, `task_obs` can be an integer denoting the task index. In other cases, `task_obs` can provide richer information.

```
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
```

## Documentation

[https://mtenv.readthedocs.io](https://mtenv.readthedocs.io)

## Contributing to MTEnv

There are several ways to contribute to MTEnv.

1. Use MTEnv in your research.

2. Contribute a new environment. We currently support [three environment suites](http://localhost:8000/pages/envs.html) via MTEnv and are looking forward to adding more environments. Contributors will be added as authors of the library. You can learn more about the workflow of adding an environment [here.](http://localhost:8000/pages/contribute_envs.html)

3. Check out the [beginner-friendly](https://github.com) issues on GitHub and contribute to fixing those issues.

4. Check out additional details [here](https://github.com/facebookresearch/mtenv/blob/main/.github/CONTRIBUTING.md).

## Community

Ask questions in the chat or github issues:
* [Chat](https://mtenv.zulipchat.com)
* [Issues](https://github.com/facebookresearch/mtenv/issues)

### Additional Links

* [Terms of Use](https://opensource.facebook.com/legal/terms)
* [Privacy Policy](https://opensource.facebook.com/legal/privacy)