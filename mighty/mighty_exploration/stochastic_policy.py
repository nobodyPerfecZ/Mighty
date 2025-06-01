from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Normal

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

                return action.detach().cpu().numpy(), logp

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
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action).unsqueeze(-1)
            return action.detach().cpu().numpy(), log_prob * self.entropy_coefficient
        else:
            # If model has attribute continuous_action=True, we know:
            #   model(state) → (action, z, mean, log_std)
            if hasattr(self.model, "continuous_action") and getattr(
                self.model, "continuous_action"
            ):
                # 1) Forward pass: get (action, z, mean, log_std)
                action, z, mean, log_std = self.model(
                    state
                )  # each: [batch, action_dim]
                std = torch.exp(log_std)  # [batch, action_dim]
                dist = Normal(mean, std)

                # 2) Compute log_prob of "z" under N(mean, std)
                log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)  # [batch, 1]

                # 3) Tanh Jacobian‐correction: sum_i log(1 − tanh(z_i)^2 + ε)
                eps = 1e-6
                log_correction = torch.log(1.0 - torch.tanh(z).pow(2) + eps).sum(
                    dim=-1, keepdim=True
                )  # [batch, 1]

                # 4) Final log_prob of a = tanh(z)
                log_prob = log_pz - log_correction  # [batch, 1]

                # 5) (Optional) multiply by entropy_coeff to get “weighted log_prob”
                weighted_log_prob = log_prob * self.entropy_coefficient

                return action.detach().cpu().numpy(), weighted_log_prob

            # If it’s actually a SACModel, fallback (should only happen in training if model∈SACModel)
            elif isinstance(self.model, SACModel):
                action, z, mean, log_std = self.model(state, deterministic=False)
                std = torch.exp(log_std)
                dist = Normal(mean, std)

                log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)
                weighted_log_prob = log_pz * self.entropy_coefficient
                return action.detach().cpu().numpy(), weighted_log_prob

            # If it’s “mean, std”‐style continuous (rare in our code), handle that case
            else:
                mean, std = self.model(state)
                dist = Normal(mean, std)
                z = dist.rsample()  # [batch, action_dim]
                action = torch.tanh(z)  # [batch, action_dim]

                log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)
                eps = 1e-6
                log_correction = torch.log(1.0 - action.pow(2) + eps).sum(
                    dim=-1, keepdim=True
                )
                log_prob = log_pz - log_correction  # [batch, 1]
                entropy = dist.entropy().sum(dim=-1, keepdim=True)  # [batch, 1]
                weighted_log_prob = log_prob * entropy

                return action.detach().cpu().numpy(), weighted_log_prob

    def forward(self, s):
        """
        Alias for explore, so policy(s) returns (action, weighted_log_prob).
        """
        return self.explore(s)
