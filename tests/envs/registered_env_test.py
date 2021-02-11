# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from mtenv import make
from mtenv.envs.registration import MultitaskEnvSpec, mtenv_registry
from tests.utils.utils import validate_mtenv

ConfigType = Dict[str, Any]


def get_env_spec() -> List[Dict[str, MultitaskEnvSpec]]:
    mtenv_env_path = os.environ.get("NOX_MTENV_ENV_PATH", "")
    if mtenv_env_path == "":
        # test all envs
        return mtenv_registry.env_specs.items()
    else:
        # test only those environments which are on NOX_MTENV_ENV_PATH

        mtenv_env_path = str(Path(mtenv_env_path).resolve())
        env_specs = deepcopy(mtenv_registry.env_specs)
        for key in list(env_specs.keys()):
            entry_point = env_specs[key].entry_point.split(":")[0].replace(".", "/")
            if mtenv_env_path not in str(Path(entry_point).resolve()):
                env_specs.pop(key)
        return env_specs.items()


def get_test_kwargs_from_spec(spec: MultitaskEnvSpec, key: str) -> List[Dict[str, Any]]:
    if spec.test_kwargs and key in spec.test_kwargs:
        return spec.test_kwargs[key]
    else:
        return []


def get_configs(get_valid_env_args: bool) -> Tuple[ConfigType, ConfigType]:
    configs = []
    key = "valid_env_kwargs" if get_valid_env_args else "invalid_env_kwargs"
    for env_name, spec in get_env_spec():
        test_config = deepcopy(spec.test_kwargs)
        for key_to_pop in ["valid_env_kwargs", "invalid_env_kwargs"]:
            if key_to_pop in test_config:
                test_config.pop(key_to_pop)
        for params in get_test_kwargs_from_spec(spec, key):
            env_config = deepcopy(params)
            env_config["id"] = env_name
            configs.append((env_config, deepcopy(test_config)))
        if get_valid_env_args:
            env_config = deepcopy(spec.kwargs)
            env_config["id"] = env_name
            configs.append((env_config, deepcopy(test_config)))
    return configs


@pytest.mark.parametrize(
    "env_config, test_config", get_configs(get_valid_env_args=True)
)
def test_registered_env_with_valid_input(env_config, test_config):
    env = make(**env_config)
    validate_mtenv(env=env, **test_config)


@pytest.mark.parametrize(
    "env_config, test_config", get_configs(get_valid_env_args=False)
)
def test_registered_env_with_invalid_input(env_config, test_config):
    with pytest.raises(Exception):
        env = make(**env_config)
        validate_mtenv(env=env, **test_config)
