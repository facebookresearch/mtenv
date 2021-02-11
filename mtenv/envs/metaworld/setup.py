# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path

import setuptools

from mtenv.utils.setup_utils import parse_dependency

env_name = "metaworld"
path = Path(__file__).parent / "requirements.txt"
requirements = parse_dependency(path)


with (Path(__file__).parent / "README.md").open() as fh:
    long_description = fh.read()

setuptools.setup(
    name=env_name,
    version="0.0.1",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
