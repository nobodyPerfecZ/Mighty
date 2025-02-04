from typing import Dict

from .mighty_maml_runner import MightyMAMLRunner, MightyTRPOMAMLRunner
from .mighty_online_runner import MightyOnlineRunner
from .mighty_runner import MightyRunner

VALID_RUNNER_TYPES = ["standard", "default", "online"]
RUNNER_CLASSES: Dict[str, type[MightyRunner]] = {
    "standard": MightyOnlineRunner,
    "default": MightyOnlineRunner,
    "online": MightyOnlineRunner,
}

try:
    import evosax  # noqa: F401

    found = True
except ImportError:
    print("evosax not found, to use ES runners please install mighty[es].")
    found = False

if found:
    from .mighty_es_runner import MightyESRunner

    VALID_RUNNER_TYPES.append("es")
    RUNNER_CLASSES["es"] = MightyESRunner


from .factory import get_runner_class  # noqa: E402

__all__ = [
    "MightyRunner",
    "MightyOnlineRunner",
    "MightyMAMLRunner",
    "MightyTRPOMAMLRunner",
    "MightyESRunner",
    "get_runner_class",
]
