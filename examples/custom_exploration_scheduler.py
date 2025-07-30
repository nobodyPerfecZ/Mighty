"""Epsilon Greedy Scheduler."""

from __future__ import annotations

from mighty.mighty_meta.mighty_component import MightyMetaComponent


class EpsilonSchedule(MightyMetaComponent):
    """Cosine LR Schedule with optional warm restarts."""

    def __init__(
        self, initial_epsilon=1.0, num_decay_steps=40000, target_epsilon=0.01
    ) -> None:
        """Epsilon schedule initialization.

        :param initial_epsilon: Initial maximal epsilon
        :param num_decay_steps: Length of schedule in steps
        :param target_epsilon: Minimal epsilon
        :return:
        """
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.target_epsilon = target_epsilon
        self.num_decay_steps = num_decay_steps
        self.pre_step_methods = [self.adapt_epsilon]

    def adapt_epsilon(self, metrics):
        """Adapt epsilon on step.

        :param metrics: Dict of current metrics
        :return:
        """
        current_epsilon = self.initial_epsilon - (
            (self.initial_epsilon - self.target_epsilon)
            * metrics["step"]
            / self.num_decay_steps
        )
        metrics["hp/pi_epsilon"] = current_epsilon
