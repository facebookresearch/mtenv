# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
import base64
import os
from pathlib import Path
from typing import List, Set

import nox
from nox.sessions import Session

DEFAULT_PYTHON_VERSIONS = ["3.6", "3.7", "3.8", "3.9"]

PYTHON_VERSIONS = os.environ.get(
    "NOX_PYTHON_VERSIONS", ",".join(DEFAULT_PYTHON_VERSIONS)
).split(",")


def setup_env(session: Session, name: str) -> None:
    env = {}
    if name in ["metaworld"]:
        key = "CIRCLECI_MJKEY"
        if key in os.environ:
            # job is running in CI
            env[
                "LD_LIBRARY_PATH"
            ] = "$LD_LIBRARY_PATH:/home/circleci/.mujoco/mujoco200/bin"
    session.install(f".[{name}]", env=env)


def setup_mtenv(session: Session) -> None:
    key = "CIRCLECI_MJKEY"
    if key in os.environ:
        # job is running in CI
        mjkey = base64.b64decode(os.environ[key]).decode("utf-8")
        mjkey_path = "/home/circleci/.mujoco/mjkey.txt"
        with open(mjkey_path, "w") as f:
            # even if the mjkey exists, we can safely overwrite it.
            for line in mjkey:
                f.write(line)
    session.install("--upgrade", "setuptools", "pip")
    session.install(".[dev]")


def get_core_paths(root: str) -> List[str]:
    """Return all the files/directories that are part of core package.

    In practice, it just excludes the directories in env module"""
    paths = []
    for _path in Path(root).iterdir():
        if _path.stem == "envs":
            for _env_path in _path.iterdir():
                if _env_path.is_file():
                    paths.append(str(_env_path))
        else:
            paths.append(str(_path))
    return paths


class EnvSetup:
    def __init__(
        self, name: str, setup_path: Path, supported_python_versions: Set[str]
    ) -> None:
        self.name = name
        self.setup_path = str(setup_path)
        self.path = str(setup_path.parent)
        self.supported_python_versions = supported_python_versions


def parse_setup_file(session: Session, setup_path: Path) -> EnvSetup:
    command = ["python", str(setup_path), "--name", "--classifiers"]
    classifiers = session.run(*command, silent=True).splitlines()
    name = classifiers[0]
    python_version_string = "Programming Language :: Python :: "
    supported_python_versions = {
        stmt.replace(python_version_string, "")
        for stmt in classifiers[1:]
        if python_version_string in stmt
    }
    return EnvSetup(
        name=name,
        setup_path=setup_path,
        supported_python_versions=supported_python_versions,
    )


def get_all_envsetups(session: Session) -> List[EnvSetup]:
    return [
        parse_setup_file(session=session, setup_path=setup_path)
        for setup_path in Path("mtenv/envs").glob("**/setup.py")
    ]


def get_all_env_setup_paths_as_nox_params():
    return [
        nox.param(setup_path, id=setup_path.parent.stem)
        for setup_path in Path("mtenv/envs").glob("**/setup.py")
    ]


def get_supported_envsetups(session: Session) -> List[EnvSetup]:
    """Get the list of EnvSetups that can run in a given session."""
    return [
        env_setup
        for env_setup in get_all_envsetups(session=session)
        if session.python in env_setup.supported_python_versions
    ]


def get_supported_env_paths(session: Session) -> List[str]:
    """Get the list of env_paths that can run in a given session."""
    return [env_setup.path for env_setup in get_supported_envsetups(session=session)]


@nox.session(python=PYTHON_VERSIONS)
def lint(session: Session) -> None:
    setup_mtenv(session=session)
    for _path in (
        get_core_paths(root="mtenv")
        + get_core_paths(root="tests")
        + get_supported_env_paths(session=session)
    ):
        session.run("black", "--check", _path)
        session.run("flake8", _path)


@nox.session(python=PYTHON_VERSIONS)
def mypy(session: Session) -> None:
    setup_mtenv(session=session)
    for _path in get_core_paths(root="mtenv"):
        session.run("mypy", "--strict", _path)
    for envsetup in get_supported_envsetups(session=session):
        setup_env(session=session, name=envsetup.name)
        session.run("mypy", envsetup.path)


@nox.session(python=PYTHON_VERSIONS)
def test_wrappers(session) -> None:
    setup_mtenv(session=session)
    session.run("pytest", "tests/wrappers")


@nox.session(python=PYTHON_VERSIONS)
def test_examples(session) -> None:
    setup_mtenv(session=session)
    session.run("pytest", "tests/examples")


@nox.session(python=PYTHON_VERSIONS)
@nox.parametrize("env_setup_path", get_all_env_setup_paths_as_nox_params())
def test_envs(session, env_setup_path) -> None:
    setup_mtenv(session=session)

    envsetup = parse_setup_file(session=session, setup_path=env_setup_path)

    if session.python not in envsetup.supported_python_versions:
        print(f"Python {session.python} is not supported  by {envsetup.name}")
        return
    setup_env(session=session, name=envsetup.name)
    env = {"NOX_MTENV_ENV_PATH": envsetup.path}
    command_for_headless_rendering = [
        "xvfb-run",
        "-a",
        "-s",
        "-screen 0 1024x768x24 -ac +extension GLX +render -noreset",
    ]
    commands = []
    key = "CIRCLECI_MJKEY"
    if key in os.environ and envsetup.name in ["metaworld"]:
        env["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/home/circleci/.mujoco/mujoco200/bin"
    if envsetup.name.startswith("MT-HiPBMDP"):
        env["PYTHONPATH"] = "mtenv/envs/hipbmdp/local_dm_control_suite"
    if envsetup.name in ["hipbmdp", "mpte"]:
        commands = commands + command_for_headless_rendering
    commands = commands + ["pytest", "tests/envs"]
    session.run(*commands, env=env)
