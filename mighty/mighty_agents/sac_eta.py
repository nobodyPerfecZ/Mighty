from pathlib import Path
from typing import Optional

from mighty.mighty_agents.sac import MightySACAgent
from mighty.mighty_update import SACEtaUpdate
from mighty.mighty_utils.mighty_types import MIGHTYENV


class MightySACEtaAgent(MightySACAgent):
    """SAC agent with a learnable risk parameter eta (port of rejax's SACEta).

    Extends SAC with a `target_risk_ratio` hyperparameter and an additional
    learnable parameter eta that is adapted during training, see SACEtaUpdate.
    """

    def __init__(
        self,
        output_dir: Path,
        env: MIGHTYENV,
        eval_env: Optional[MIGHTYENV] = None,
        target_risk_ratio: float = 0.1,
        eta_lr: float = 3e-4,
        **kwargs,
    ):
        self.target_risk_ratio = target_risk_ratio
        self.eta_lr = eta_lr

        super().__init__(output_dir=output_dir, env=env, eval_env=eval_env, **kwargs)

        self.loss_buffer["Update/eta_loss"] = []
        self.loss_buffer["Update/eta_value"] = []

    def _initialize_agent(self) -> None:
        super()._initialize_agent()

        # Replace the plain SAC updater with the eta-augmented one
        self.update_fn = SACEtaUpdate(
            model=self.model,
            policy_lr=self.policy_lr,
            q_lr=self.q_lr,
            tau=self.tau,
            alpha=self.alpha,
            gamma=self.gamma,
            auto_alpha=self.auto_alpha,
            target_entropy=self.target_entropy,
            alpha_lr=self.alpha_lr,
            policy_frequency=self.policy_frequency,
            target_network_frequency=self.target_network_frequency,
            target_risk_ratio=self.target_risk_ratio,
            eta_lr=self.eta_lr,
        )
