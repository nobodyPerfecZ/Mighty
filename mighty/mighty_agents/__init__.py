from .base_agent import MightyAgent
from .dqn import MightyDQNAgent
from .ppo import MightyPPOAgent
from .sac import MightySACAgent
from .sac_eta import MightySACEtaAgent

VALID_AGENT_TYPES = ["DQN", "PPO", "SAC", "SACEta", "DDQN"]
AGENT_CLASSES = {
    "DQN": MightyDQNAgent,
    "PPO": MightyPPOAgent,
    "SAC": MightySACAgent,
    "SACEta": MightySACEtaAgent,
    "DDQN": MightyDQNAgent,
}

from .factory import get_agent_class  # noqa: E402

__all__ = [
    "MightyAgent",
    "get_agent_class",
    "MightyDQNAgent",
    "MightyPPOAgent",
    "MightySACAgent",
    "MightySACEtaAgent",
]
