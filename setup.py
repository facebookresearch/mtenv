# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
import codecs
import os.path
import subprocess
from pathlib import Path

import setuptools


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


def parse_dependency(filepath):
    dep_list = []
    for dep in open(filepath).read().splitlines():
        if dep.startswith("#"):
            continue
        key = "#egg="
        if key in dep:
            git_link, egg_name = dep.split(key)
            dep = f"{egg_name} @ {git_link}"
        dep_list.append(dep)
    return dep_list


base_requirements = parse_dependency("requirements/base.txt")
dev_requirements = base_requirements + parse_dependency("requirements/dev.txt")


extras_require = {}

for setup_path in Path("mtenv/envs").glob("**/setup.py"):
    env_path = setup_path.parent
    env_name = (
        subprocess.run(["python", setup_path, "--name"], stdout=subprocess.PIPE)
        .stdout.decode()
        .strip()
    )
    extras_require[env_name] = base_requirements + parse_dependency(
        f"{str(env_path)}/requirements.txt"
    )

extras_require["all"] = list(
    set([dep for requirements in extras_require.values() for dep in requirements])
)
extras_require["dev"] = dev_requirements

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtenv",
    version=get_version("mtenv/__init__.py"),
    author="Shagun Sodhani, Ludovic Denoyer, Pierre-Alexandre Kamienny, Olivier Delalleau",
    author_email="sshagunsodhani@gmail.com, denoyer@fb.com, pakamienny@fb.com, odelalleau@fb.com",
    description="MTEnv: MultiTask Environments for Reinforcement Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=base_requirements,
    url="https://github.com/facbookresearch/mtenv",
    packages=setuptools.find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests", "docs", "docsrc"]
    ),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    extras_require=extras_require,
)
