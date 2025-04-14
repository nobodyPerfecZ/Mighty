"""Type helpers for the mighty package."""

from __future__ import annotations

import importlib
from typing import Any, NewType

import hydra
from omegaconf import DictConfig

TypeKwargs = NewType("TypeKwargs", dict[str, Any] | DictConfig)

MIGHTYENV = None

dacbench = importlib.util.find_spec("dacbench")
dacbench_found = dacbench is not None
if dacbench_found:
    import dacbench

    MIGHTYENV = dacbench.AbstractEnv
    DACENV = dacbench.AbstractEnv
else:
    DACENV = int

carl = importlib.util.find_spec("carl")
carl_found = carl is not None
if carl_found:
    from carl.envs.carl_env import CARLEnv

    if MIGHTYENV is None:
        MIGHTYENV = CARLEnv
    CARLENV = CARLEnv
else:
    CARLENV = int

if not carl_found and not dacbench_found:
    import gymnasium as gym

    MIGHTYENV = gym.Env


def retrieve_class(cls: str | DictConfig | type, default_cls: type) -> type:
    """Get mighty class."""
    if cls is None:
        cls = default_cls
    elif isinstance(cls, DictConfig):
        cls = hydra.utils.get_class(cls._target_)
    elif isinstance(cls, str):
        cls = hydra.utils.get_class(cls)
    return cls
