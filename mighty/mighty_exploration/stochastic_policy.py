from __future__ import annotations

import torch
from typing import Tuple
import numpy as np

from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy
from mighty.mighty_models import SACModel


class StochasticPolicy(MightyExplorationPolicy):
    """Entropy-Based Exploration for discrete and continuous action spaces."""

    def __init__(
        self, algo, model, entropy_coefficient: float = 0.2, discrete: bool = True
    ):
        """
        :param algo: the RL algorithm instance
        :param model: the policy model
        :param entropy_coefficient: weight on entropy term
        :param discrete: whether the action space is discrete
        """
        super().__init__(algo, model, discrete)
        self.entropy_coefficient = entropy_coefficient
        self.discrete = discrete

        # --- override sample_action only for continuous SAC ---
        if not discrete and isinstance(model, SACModel):
            # for evaluation use deterministic=True; training will go through .explore()
            def _sac_sample(state_np):
                state = torch.as_tensor(state_np, dtype=torch.float32)
                # forward returns (action, z, mean, log_std)
                action, z, mean, log_std = model(state, deterministic=True)
                logp = model.policy_log_prob(z, mean, log_std)
                return action, logp

            self.sample_action = _sac_sample

    def explore(self, s, return_logp, metrics=None) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Given observations `s`, sample an exploratory action and compute a weighted log-prob.

        Returns:
          action: numpy array of actions
          weighted_log_prob: Tensor of shape [batch, 1]
        """
        state = torch.as_tensor(s, dtype=torch.float32)
        if self.discrete:
            logits = self.model(state)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
        else:
            # Model returns: action (tanh-squashed), z (pre-tanh), mean, log_std
            action, z, mean, log_std = self.model(state)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            # log_prob and entropy over pre-tanh sample
            log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        # Weighted by entropy coefficient
        weighted_log_prob = log_prob * self.entropy_coefficient
        return action.detach().cpu().numpy(), weighted_log_prob

    def forward(self, s):
        """
        Alias for explore, so policy(s) returns (action, weighted_log_prob).
        """
        return self.explore(s)
