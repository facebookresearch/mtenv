# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from pathlib import Path
from typing import List


def parse_dependency(filepath: Path) -> List[str]:
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
