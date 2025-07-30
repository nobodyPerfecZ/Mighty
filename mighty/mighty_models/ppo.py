import math
from typing import Tuple

import torch
import torch.nn as nn

from mighty.mighty_models.networks import ACTIVATIONS, make_feature_extractor


class PPOModel(nn.Module):
    """PPO Model with policy and value networks."""

    output_style = (
        "squashed_gaussian"  # For continuous actions, we use squashed Gaussian output
    )

    def __init__(
        self,
        obs_shape: int,
        action_size: int,
        continuous_action: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        tanh_squash: bool = False,  # NEW: Toggle between tanh squashing and standard PPO
        **kwargs,
    ):
        """Initialize the PPO model."""
        super().__init__()

        self.obs_size = int(obs_shape)
        self.action_size = int(action_size)
        self.continuous_action = continuous_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash

        # Extract configuration from kwargs or use defaults
        head_kwargs = kwargs.get(
            "head_kwargs",
            {"hidden_sizes": [64], "layer_norm": True, "activation": "tanh"},
        )
        feature_extractor_kwargs = kwargs.get(
            "feature_extractor_kwargs",
            {
                "obs_shape": self.obs_size,
                "activation": "tanh",
                "hidden_sizes": [64, 64],
                "n_layers": 2,
            },
        )

        # Allow direct specification of hidden_sizes and activation at top level
        if "hidden_sizes" in kwargs:
            feature_extractor_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
        if "activation" in kwargs:
            feature_extractor_kwargs["activation"] = kwargs["activation"]
            head_kwargs["activation"] = kwargs["activation"]

        # Make feature extractors
        self.feature_extractor_policy, feat_dim = make_feature_extractor(
            **feature_extractor_kwargs
        )
        self.feature_extractor_value, _ = make_feature_extractor(
            **feature_extractor_kwargs
        )

        if self.continuous_action:
            if self.tanh_squash:
                # Tanh squashing mode: output mean + log_std from network
                final_out_dim = action_size * 2
                # No learnable parameter needed
                self.log_std = None
            else:
                # Standard PPO mode: output only mean, use learnable log_std parameter
                final_out_dim = action_size
                self.log_std = nn.Parameter(torch.zeros(action_size))
        else:
            # For discrete actions, output logits of size = action_size
            final_out_dim = action_size

        # (Architecture based on
        # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py)

        # Policy network
        self.policy_head = make_ppo_head(feat_dim, final_out_dim, **head_kwargs)

        # Value network
        self.value_head = make_ppo_head(feat_dim, 1, **head_kwargs)

        # Orthogonal initialization
        def _init_weights(m: nn.Module):
            if isinstance(m, nn.Linear):
                out_dim = m.out_features
                if self.continuous_action and out_dim == final_out_dim:
                    # This is the final policy‐output layer (mean & log_std):
                    gain = 0.01
                elif (not self.continuous_action) and out_dim == action_size:
                    # Final policy‐output layer (discrete‐logits):
                    gain = 0.01
                elif out_dim == 1:
                    # Final value‐output layer:
                    gain = 1.0
                else:
                    # Any intermediate hidden layer:
                    gain = math.sqrt(2)
                nn.init.orthogonal_(m.weight, gain)
                nn.init.constant_(m.bias, 0.0)

        self.apply(_init_weights)

        # Create a value function wrapper that can be called like a module
        class ValueFunctionWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent_model = parent_model

            def forward(self, x):
                return self.parent_model.forward_value(x)

        self.value_function_module = ValueFunctionWrapper(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.

        Returns:
        - If discrete: logits tensor
        - If continuous + tanh_squash: (action, z, mean, log_std)
        - If continuous + not tanh_squash: (action, mean, log_std)
        """

        if self.continuous_action:
            if self.tanh_squash:
                # TANH SQUASHING MODE (4-tuple return)
                feats = self.feature_extractor_policy(x)
                raw = self.policy_head(feats)  # [batch, 2 * action_size]
                mean, log_std = raw.chunk(2, dim=-1)  # each [batch, action_size]
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)  # [batch, action_size]

                # Sample a raw Gaussian z for reparameterization
                eps = torch.randn_like(mean)
                z = mean + std * eps  # [batch, action_size]
                action = torch.tanh(z)  # squash to [−1, +1]

                return action, z, mean, log_std

            else:
                # STANDARD PPO MODE (3-tuple return)
                feats = self.feature_extractor_policy(x)
                mean = self.policy_head(feats)  # [batch, action_size]

                # Use the learnable log_std parameter
                log_std = self.log_std.expand_as(mean)  # [batch, action_size]
                log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
                std = torch.exp(log_std)  # [batch, action_size]

                # Sample directly from Normal distribution (NO TANH)
                eps = torch.randn_like(mean)
                action = mean + std * eps  # [batch, action_size] - direct sampling

                return action, mean, log_std

        else:
            # DISCRETE ACTION MODE
            feats = self.feature_extractor_policy(x)
            logits = self.policy_head(feats)  # [batch, action_size]
            return logits

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value network."""
        feats = self.feature_extractor_value(x)
        result = self.value_head(feats)  # [batch, 1]
        return result


def make_ppo_head(
    in_size, outsize, hidden_sizes=None, layer_norm=True, activation="relu"
):
    """Make PPO head network."""

    # Make fully connected layers
    if hidden_sizes is None:
        hidden_sizes = []

    layers = []
    last_size = in_size
    if isinstance(last_size, list):
        last_size = last_size[0]

    for size in hidden_sizes:
        layers.append(nn.Linear(last_size, size))
        if layer_norm:
            layers.append(nn.LayerNorm(size))
        layers.append(ACTIVATIONS[activation]())
        last_size = size
    layers.append(nn.Linear(last_size, outsize))

    return nn.Sequential(*layers)
