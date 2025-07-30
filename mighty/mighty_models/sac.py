from typing import Tuple

import torch
from torch import nn

from mighty.mighty_models.networks import ACTIVATIONS, make_feature_extractor


class SACModel(nn.Module):
    """SAC Model with squashed Gaussian policy and twin Q-networks."""

    output_style = (
        "squashed_gaussian"  # For continuous actions, we use squashed Gaussian output
    )

    def __init__(
        self,
        obs_size: int,
        action_size: int,
        log_std_min: float = -20,
        log_std_max: float = 2,
        **kwargs,
    ):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # This model is continuous only
        self.continuous_action = True

        head_kwargs = {"hidden_sizes": [256, 256], "activation": "relu"}
        feature_extractor_kwargs = {
            "obs_shape": self.obs_size,
            "activation": "relu",
            "hidden_sizes": [256, 256],
            "n_layers": 2,
        }
        
        # Allow direct specification of hidden_sizes and activation at top level
        if "hidden_sizes" in kwargs:
            feature_extractor_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
            head_kwargs["hidden_sizes"] = kwargs["hidden_sizes"]
        if "activation" in kwargs:
            feature_extractor_kwargs["activation"] = kwargs["activation"]
            head_kwargs["activation"] = kwargs["activation"]
            
        if "head_kwargs" in kwargs:
            head_kwargs.update(kwargs["head_kwargs"])
        if "feature_extractor_kwargs" in kwargs:
            feature_extractor_kwargs.update(kwargs["feature_extractor_kwargs"])

        # Store for Q-network creation
        self.hidden_sizes = feature_extractor_kwargs.get("hidden_sizes", [256, 256])
        self.activation = feature_extractor_kwargs.get("activation", "relu")

        # Shared feature extractor for policy
        self.feature_extractor, out_dim = make_feature_extractor(
            **feature_extractor_kwargs
        )

        # Policy network outputs mean and log_std
        self.policy_net = nn.Linear(out_dim, action_size * 2)

        # Twin Q-networks
        # — live Q-nets —
        self.q_net1 = make_q_head(
            in_size=self.obs_size + self.action_size, **head_kwargs
        )
        self.q_net2 = make_q_head(
            in_size=self.obs_size + self.action_size, **head_kwargs
        )

        # Target Q-networks
        self.target_q_net1 = make_q_head(
            in_size=self.obs_size + self.action_size, **head_kwargs
        )
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2 = make_q_head(
            in_size=self.obs_size + self.action_size, **head_kwargs
        )
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())

        # Freeze target networks
        for p in self.target_q_net1.parameters():
            p.requires_grad = False
        for p in self.target_q_net2.parameters():
            p.requires_grad = False

        # Create a value function wrapper for compatibility
        class ValueFunctionWrapper(nn.Module):
            def __init__(self, parent_model):
                super().__init__()
                self.parent_model = parent_model
                
            def forward(self, x):
                # SAC doesn't have a separate value function, but for compatibility
                # we can return the minimum of the two Q-values with a zero action
                # This is mainly for interface compatibility
                batch_size = x.shape[0]
                zero_action = torch.zeros(batch_size, self.parent_model.action_size, device=x.device)
                state_action = torch.cat([x, zero_action], dim=-1)
                q1 = self.parent_model.forward_q1(state_action)
                q2 = self.parent_model.forward_q2(state_action)
                return torch.min(q1, q2)
        
        self.value_function_module = ValueFunctionWrapper(self)

    def forward(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy sampling.

        Returns:
          action: torch.Tensor in [-1,1]
          z: raw Gaussian sample before tanh
          mean: Gaussian mean
          log_std: Gaussian log std
        """
        feats = self.feature_extractor(state)
        x = self.policy_net(feats)
        mean, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z)
        return action, z, mean, log_std

    def policy_log_prob(
        self, z: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log-prob of action a = tanh(z), correcting for tanh transform.
        """
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_pz = dist.log_prob(z).sum(dim=-1, keepdim=True)
        eps = 1e-6  # small constant to avoid numerical issues
        log_correction = (torch.log(1 - torch.tanh(z).pow(2) + eps)).sum(
            dim=-1, keepdim=True
        )
        log_pa = log_pz - log_correction
        return log_pa

    def forward_q1(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net1(state_action)

    def forward_q2(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.q_net2(state_action)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the value function for compatibility."""
        return self.value_function_module(x)


def make_q_head(in_size, hidden_sizes=None, activation="relu"):
    """Make Q head network."""
    # Make fully connected layers
    if hidden_sizes is None:
        hidden_sizes = []

    layers = []
    last_size = in_size
    if isinstance(last_size, list):
        last_size = last_size[0]

    for size in hidden_sizes:
        layers.append(nn.Linear(last_size, size))
        layers.append(ACTIVATIONS[activation]())
        last_size = size

    layers.append(nn.Linear(last_size, 1))
    return nn.Sequential(*layers)