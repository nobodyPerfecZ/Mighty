from __future__ import annotations

import numpy as np
import torch

from mighty.mighty_replay.mighty_replay_buffer import MightyReplay, TransitionBatch


class PrioritizedReplay(MightyReplay):
    """Prioritized Replay Buffer."""

    def __init__(
        self,
        capacity,
        alpha=1.0,
        beta=1.0,
        epsilon=1e-4,
        device: torch.device | str = "cpu",
        keep_infos=False,
        flatten_infos=False,
    ):
        """Initialize Buffer.

        :param capacity: Buffer size
        :param alpha: Priorization exponent
        :param beta: Bias exponent
        :param epsilon: Step size
        :param random_seed: Seed for sampling
        :param keep_infos: Keep the extra info dict. Required for some algorithms.
        :param flatten_infos: Make flat list from infos.
            Might be necessary, depending on info content.
        :return:
        """
        super().__init__(capacity, keep_infos, flatten_infos, device)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def add(self, transition_batch, metrics):
        """Add transition(s).

        :param transition_batch: Transition(s) to add
        :param metrics: Current metrics dict
        :return:
        """
        super().add(transition_batch, metrics)
        advantage = metrics["td_error"]
        advantage = np.power(np.abs(advantage) + self.epsilon, self.alpha)
        if len(self.advantages) == 0:
            self.advantages = torch.from_numpy(advantage)
        else:
            self.advantages = torch.cat((self.advantages, torch.from_numpy(advantage)))
        while len(self.advantages) > self.capacity:
            self.advantages.pop(0)

    def reset(self):
        """Reset the buffer."""
        super().reset()
        self.advantages = []

    def sample(self, batch_size=32):
        """Sample transitions."""
        probabilities = np.array(self.advantages) / sum(self.advantages)
        sample_weights = np.power(probabilities * len(self), -self.beta)
        sample_weights /= sample_weights.max()
        normalizer = 1 / sum(sample_weights)
        sample_weights = np.array([x * normalizer for x in sample_weights])
        # Get rid of rounding errors
        sample_weights[-1] = max(0, 1 - np.sum(sample_weights[0:-1]))

        batch_indices = self.rng.choice(
            np.arange(len(self)), size=batch_size, p=sample_weights
        )
        return TransitionBatch(
            self.obs[batch_indices],
            self.actions[batch_indices],
            self.rewards[batch_indices],
            self.next_obs[batch_indices],
            self.dones[batch_indices],
            device=self.device,
        )
