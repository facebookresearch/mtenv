# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, Tuple, Union

import numpy as np

TaskObsType = Union[str, int, float, np.ndarray]
ActionType = Union[str, int, float, np.ndarray]
EnvObsType = Union[np.ndarray]
ObsType = Dict[str, Union[EnvObsType, TaskObsType]]
RewardType = float
DoneType = bool
InfoType = Dict[str, Any]
StepReturnType = Tuple[ObsType, RewardType, DoneType, InfoType]
EnvStepReturnType = Tuple[EnvObsType, RewardType, DoneType, InfoType]
TaskStateType = Any
