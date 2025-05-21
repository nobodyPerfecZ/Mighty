from typing import Tuple

import torch
import torch.nn as nn
import math

from mighty.mighty_models.networks import make_feature_extractor


class PPOModel(nn.Module):
    """PPO Model with policy and value networks."""

    def __init__(
        self,
        obs_shape,
        action_size,
        hidden_sizes=[64, 64],
        activation="tanh",
        continuous_action=False,
    ):
        """Initialize the PPO model."""
        super().__init__()

        self.obs_size = int(obs_shape)
        self.action_size = int(action_size)
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.continuous_action = continuous_action

        # Make feature extractor
        self.feature_extractor_policy, self.output_size = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_shape,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        
        self.feature_extractor_value, _ = make_feature_extractor(
            architecture="mlp",
            obs_shape=obs_shape,
            n_layers=len(hidden_sizes),
            hidden_sizes=hidden_sizes,
            activation=activation,
        )
        
        
        
        
        # (Architecture based on
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py)

        # Policy network
        self.policy_head = nn.Sequential(
            self.feature_extractor_policy,
            nn.Linear(self.output_size, 64),
            nn.Linear(
                hidden_sizes[0],
                2 if continuous_action else action_size
            ),
        )

        # Value network
        self.value_head = nn.Sequential(
            self.feature_extractor_value,
            nn.Linear(self.output_size, 64),
            nn.Linear(hidden_sizes[0], 1),
        )
        
        # Orthogonal initialization
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                # Set gain: policy output small, value output 1, hidden layers sqrt(2)
                if m.out_features == (2 if continuous_action else action_size):
                    gain = 0.01
                elif m.out_features == 1:
                    gain = 1.0
                else:
                    gain = math.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the policy network."""

        x = self.policy_head(x)

        if self.continuous_action:
            mean, log_std = x.chunk(2, dim=-1)
            # FIXME: the clamping is hardcoded here, should be a probabyl be a hyperparameter
            log_std = log_std.clamp(-20, 2)  # Remove the extra dimension
            return mean, log_std.exp()
        else:
            return x  # return logits

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        return self.value_head(x)