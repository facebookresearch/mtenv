# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

import copy
import importlib
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

from gym import error, logger
from gym.envs.registration import (
    EnvSpec,
    _check_spec_register,
    _check_version_exists,
    current_namespace,
    find_highest_version,
    get_env_id,
    load,
    parse_env_id,
    registry,
)
from gym.wrappers import (  # type: ignore[attr-defined]
    AutoResetWrapper,
    HumanRendering,
    OrderEnforcing,
    StepAPICompatibility,
    TimeLimit,
)
from gym.wrappers.env_checker import PassiveEnvChecker


@dataclass
class MultitaskEnvSpec(EnvSpec):

    test_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MultitaskEnvSpec({self.id})"


def register(id: str, **kwargs: Any) -> None:
    """Register an environment with gym.

    The `id` parameter corresponds to the name of the environment, with the syntax as follows:
    `(namespace)/(env_name)-v(version)` where `namespace` is optional.

    It takes arbitrary keyword arguments, which are passed to the `MultitaskEnvSpec` constructor.

    Args:
        id: The environment id
        **kwargs: arbitrary keyword arguments which are passed to the environment constructor
    """
    global registry, current_namespace
    ns, name, version = parse_env_id(id)

    ns_id: Optional[str]

    if current_namespace is not None:
        if (
            kwargs.get("namespace") is not None
            and kwargs.get("namespace") != current_namespace
        ):
            logger.warn(
                f"Custom namespace `{kwargs.get('namespace')}` is being overridden "
                f"by namespace `{current_namespace}`. If you are developing a "
                "plugin you shouldn't specify a namespace in `register` "
                "calls. The namespace is specified through the "
                "entry point package metadata."
            )
        ns_id = current_namespace
    else:
        ns_id = ns

    full_id = get_env_id(ns_id, name, version)
    spec = MultitaskEnvSpec(id=full_id, **kwargs)
    _check_spec_register(spec)
    if spec.id in registry:
        logger.warn(f"Overriding environment {spec.id}")
    registry[spec.id] = spec


def make(
    id: Union[str, MultitaskEnvSpec],
    max_episode_steps: Optional[int] = None,
    autoreset: bool = False,
    new_step_api: bool = False,
    disable_env_checker: Optional[bool] = None,
    **kwargs: Any,
) -> "MTEnv":  # type: ignore[name-defined]
    """Create an environment according to the given ID.

    Args:
        id: Name of the environment. Optionally, a module to import can be included, eg. 'module:Env-v0'
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        autoreset: Whether to automatically reset the environment after each episode (AutoResetWrapper).
        new_step_api: Whether to use old or new step API (StepAPICompatibility wrapper). Will be removed at v1.0
        disable_env_checker: If to run the env checker, None will default to the environment specification `disable_env_checker`
            (which is by default False, running the environment checker),
            otherwise will run according to this parameter (`True` = not run, `False` = run)
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist then an error is raised
    """
    if isinstance(id, MultitaskEnvSpec):
        spec_ = id
    else:
        module, id = (None, id) if ":" not in id else id.split(":")  # type: ignore[assignment]
        if module is not None:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f"{e}. Environment registration via importing a module failed. "
                    f"Check whether '{module}' contains env registration and can be imported."
                )
        spec_ = registry.get(id)  # type: ignore[assignment]

        ns, name, version = parse_env_id(id)
        latest_version = find_highest_version(ns, name)
        if (
            version is not None
            and latest_version is not None
            and latest_version > version
        ):
            logger.warn(
                f"The environment {id} is out of date. You should consider "
                f"upgrading to version `v{latest_version}`."
            )
        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_env_id(ns, name, version)
            spec_ = registry.get(new_env_id)  # type: ignore[assignment]
            logger.warn(
                f"Using the latest versioned environment `{new_env_id}` "
                f"instead of the unversioned environment `{id}`."
            )

        if spec_ is None:
            _check_version_exists(ns, name, version)
            raise error.Error(f"No registered env with id: {id}")

    assert isinstance(spec_, MultitaskEnvSpec)
    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f"{spec_.id} registered but entry_point is not specified")
    elif callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        # Assume it's a string
        env_creator = load(spec_.entry_point)

    mode = _kwargs.get("render_mode")
    apply_human_rendering = False

    # If we have access to metadata we check that "render_mode" is valid and see if the HumanRendering wrapper needs to be applied
    if mode is not None and hasattr(env_creator, "metadata"):
        assert isinstance(
            env_creator.metadata, dict  # type: ignore[attr-defined]
        ), f"Expect the environment creator ({env_creator}) metadata to be dict, actual type: {type(env_creator.metadata)}"  # type: ignore[attr-defined]

        if "render_modes" in env_creator.metadata:  # type: ignore[attr-defined]
            render_modes = env_creator.metadata["render_modes"]  # type: ignore[attr-defined]
            if not isinstance(render_modes, Sequence):
                logger.warn(
                    f"Expects the environment metadata render_modes to be a Sequence (tuple or list), actual type: {type(render_modes)}"
                )

            # Apply the `HumanRendering` wrapper, if the mode=="human" but "human" not in render_modes
            if (
                mode == "human"
                and "human" not in render_modes
                and ("single_rgb_array" in render_modes or "rgb_array" in render_modes)
            ):
                logger.warn(
                    "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                    "The HumanRendering wrapper is being applied to your environment."
                )
                apply_human_rendering = True
                if "single_rgb_array" in render_modes:
                    _kwargs["render_mode"] = "single_rgb_array"
                else:
                    _kwargs["render_mode"] = "rgb_array"
            elif mode not in render_modes:
                logger.warn(
                    f"The environment is being initialised with mode ({mode}) that is not in the possible render_modes ({render_modes})."
                )
        else:
            logger.warn(
                f"The environment creator metadata doesn't include `render_modes`, contains: {list(env_creator.metadata.keys())}"  # type: ignore[attr-defined]
            )

    try:
        env = env_creator(**_kwargs)
    except TypeError as e:
        if (
            str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
            and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {id} doesn't implement human-rendering natively. "
                "Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old "
                "rendering API, which is not supported by the HumanRendering wrapper."
            )
        else:
            raise e

    # Copies the environment creation specification and kwargs to add to the environment specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    env.unwrapped.spec = spec_

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and spec_.disable_env_checker is False
    ):
        env = PassiveEnvChecker(env)  # type: ignore[no-untyped-call]

    env = StepAPICompatibility(env, new_step_api)

    # Add the order enforcing wrapper
    if spec_.order_enforce:
        env = OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps, new_step_api)
    elif spec_.max_episode_steps is not None:
        env = TimeLimit(env, spec_.max_episode_steps, new_step_api)

    # Add the autoreset wrapper
    if autoreset:
        env = AutoResetWrapper(env, new_step_api)

    # Add human rendering wrapper
    if apply_human_rendering:
        env = HumanRendering(env)

    return env
