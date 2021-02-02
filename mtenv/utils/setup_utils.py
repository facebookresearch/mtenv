# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path
from typing import List


def parse_dependency(filepath: Path) -> List[str]:
    """Parse python dependencies from a file.

    The list of dependencies is used by `setup.py` files. Lines starting
    with "#" are ingored (useful for writing comments). In case the
    dependnecy is host using git, the url is parsed and modified to make
    suitable for `setup.py` files.


    Args:
        filepath (Path):

    Returns:
        List[str]: List of dependencies
    """
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
