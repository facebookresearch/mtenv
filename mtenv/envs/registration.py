# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

from typing import Any, Dict, Optional

from gym import error
from gym.core import Env
from gym.envs.registration import EnvRegistry, EnvSpec


class MultitaskEnvSpec(EnvSpec):  # type: ignore[misc]
    def __init__(
        self,
        id: str,
        entry_point: Optional[str] = None,
        reward_threshold: Optional[int] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        nondeterministic: bool = False,
        max_episode_steps: Optional[int] = None,
        test_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """A specification for a particular instance of the environment.
        Used to register the parameters for official evaluations.

        Args:
            id (str): The official environment ID
            entry_point (Optional[str]): The Python entrypoint of the
                environment class (e.g. module.name:Class)
            reward_threshold (Optional[int]): The reward threshold before
                the task is considered solved
            kwargs (dict): The kwargs to pass to the environment class
            nondeterministic (bool): Whether this environment is
                non-deterministic even after seeding
            max_episode_steps (Optional[int]): The maximum number of steps
                that an episode can consist of
            test_kwargs (Optional[Dict[str, Any]], optional): Dictionary
                to specify parameters for automated testing. Defaults to
                None.

        """
        super().__init__(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs,
        )
        self.test_kwargs = test_kwargs

    def __repr__(self) -> str:
        return f"MultitaskEnvSpec({self.id})"

    @property
    def kwargs(self) -> Dict[str, Any]:
        return self._kwargs  # type: ignore[no-any-return]


class MultiEnvRegistry(EnvRegistry):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def register(self, id: str, **kwargs: Any) -> None:
        if id in self.env_specs:
            raise error.Error("Cannot re-register id: {}".format(id))
        self.env_specs[id] = MultitaskEnvSpec(id, **kwargs)


# Have a global registry
mtenv_registry = MultiEnvRegistry()


def register(id: str, **kwargs: Any) -> None:
    return mtenv_registry.register(id, **kwargs)


def make(id: str, **kwargs: Any) -> Env:
    env = mtenv_registry.make(id, **kwargs)
    assert isinstance(env, Env)
    return env


def spec(id: str) -> MultitaskEnvSpec:
    spec = mtenv_registry.spec(id)
    assert isinstance(spec, MultitaskEnvSpec)
    return spec
