"""Internal reward via RND."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn, optim

from mighty.mighty_meta.mighty_component import MightyMetaComponent


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


DEFAULT_NETWORK_CONFIG = [
    {"type": "Linear", "kwargs": {"out_features": 64}},
    {"type": "ReLU", "kwargs": {}},
    {"type": "Linear", "kwargs": {}},
]


class RNDNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_config: dict,
        num_additional_predictor_layers: int = 2,
    ) -> None:
        super().__init__()

        if len(input_size) == 1:
            self.input_size = input_size[0]
        else:
            self.input_size = input_size
        self.output_size = output_size

        if network_config is None or len(network_config) == 0:
            network_config = DEFAULT_NETWORK_CONFIG
        else:
            network_config = OmegaConf.to_container(network_config)

        predictor_layers = self.make_network(network_config)
        for _ in range(num_additional_predictor_layers):
            predictor_layers.append(nn.ReLU())
            predictor_layers.append(
                layer_init(nn.Linear(self.output_size, self.output_size))
            )

        # Prediction network
        self.predictor = nn.Sequential(*predictor_layers)

        # Target network
        self.target = nn.Sequential(*self.make_network(network_config))

        # target network is not trainable
        for param in self.target.parameters():
            param.requires_grad = False

    def make_network(self, network_config):
        layers = []
        input_size = self.input_size
        dummy_input = torch.zeros(self.input_size)
        for i, layer in enumerate(network_config):
            layer_type = layer["type"]
            layer_kwargs = layer["kwargs"]
            if layer_type == "Linear":
                if i == len(network_config) - 1:
                    layer_kwargs["out_features"] = self.output_size
                layer_kwargs["in_features"] = input_size
                input_size = layer_kwargs["out_features"]
            if layer_type == "Flatten":
                dummy_network = nn.Sequential(*layers)
                input_size = dummy_network(dummy_input).shape
                del dummy_network
            if layer_type in ["Linear", "Conv2d", "ConvTranspose2d", "Conv3D"]:
                layers.append(layer_init(getattr(nn, layer_type)(**layer_kwargs)))
            else:
                layers.append(getattr(nn, layer_type)(**layer_kwargs))
        return layers

    def forward(self, x):
        target = self.target(torch.tensor(x).float())
        prediction = self.predictor(torch.tensor(x).float())
        return prediction, target

    def get_error(self, x):
        prediction, target = self.forward(torch.tensor(x).float())
        error = nn.MSELoss(reduction="none")(prediction, target).mean(-1)
        return error.detach().cpu().numpy()


class RND(MightyMetaComponent):
    """Cosine LR Schedule with optional warm restarts."""

    def __init__(
        self,
        rnd_output_dim: int = 512,
        rnd_network_config: dict | None = None,
        internal_reward_weight: float = 0.1,
        rnd_lr: float = 0.001,
        rnd_eps: float = 1e-5,
        rnd_weight_decay: float = 0.01,
        update_proportion: float = 0.5,
    ) -> None:
        """Cosine schedule initialization.

        :param initial_lr: Initial maximal LR
        :param num_decay_steps: Length of schedule in steps
        :param min_lr: Minimal LR
        :param restart_every: Restart frequency
        :param restart multiplier: Multiplies current learning rate on restart.
        :return:
        """
        if rnd_network_config is None:
            rnd_network_config = {}
        super().__init__()
        self.update_proportion = update_proportion
        self.internal_reward_weight = internal_reward_weight
        self.post_step_methods = [self.get_reward]
        self.post_update_methods = [self.update_predictor]
        self.rnd_network_config = rnd_network_config
        self.rnd_output_dim = rnd_output_dim
        self.rnd_lr = rnd_lr
        self.rnd_eps = rnd_eps
        self.rnd_weight_decay = rnd_weight_decay
        self.rnd_net = None
        self.post_update_methods = [self.update_predictor]
        self.post_step_methods = [self.get_reward]

    def initialize_networks(self, input_size):
        self.rnd_net = RNDNetwork(
            input_size, self.rnd_output_dim, self.rnd_network_config
        )
        self.rnd_optimizer = optim.Adam(
            self.rnd_net.parameters(),
            lr=self.rnd_lr,
            eps=self.rnd_eps,
            weight_decay=self.rnd_weight_decay,
        )

    def get_reward(self, metrics):
        """Adapt LR on step.

        :param metrics: Dict of current metrics
        :return:
        """
        if self.rnd_net is None:
            self.initialize_networks(metrics["transition"]["next_state"].shape[1:])

        rnd_error = self.rnd_net.get_error(metrics["transition"]["next_state"])
        metrics["transition"]["intrinsic_reward"] = (
            self.internal_reward_weight * rnd_error
        )
        metrics["transition"]["reward"] = (
            metrics["transition"]["reward"] + self.internal_reward_weight * rnd_error
        )
        return metrics

    def update_predictor(self, metrics):
        for batch in metrics["update_batches"]:
            if len(batch.observations.shape) == 1:
                obs = batch.observations.unsqueeze(1)
            else:
                obs = batch.observations
            prediction, target = self.rnd_net(obs)
            loss = nn.MSELoss(reduction="none")(prediction, target.detach()).mean(-1)
            mask = torch.rand(loss.shape)
            mask = mask < self.update_proportion
            loss = (loss * mask).sum() / torch.max(mask.sum(), torch.tensor([1]))
            self.rnd_optimizer.zero_grad()
            loss.backward()
            self.rnd_optimizer.step()


class NovelD(RND):
    def __init__(
        self,
        rnd_output_dim: int = 512,
        rnd_network_config: dict | None = None,
        internal_reward_weight: float = 0.1,
        rnd_lr: float = 0.001,
        rnd_eps: float = 1e-5,
        rnd_weight_decay: float = 0.01,
        update_proportion: float = 0.5,
    ) -> None:
        """Cosine schedule initialization.

        :param initial_lr: Initial maximal LR
        :param num_decay_steps: Length of schedule in steps
        :param min_lr: Minimal LR
        :param restart_every: Restart frequency
        :param restart multiplier: Multiplies current learning rate on restart.
        :return:
        """
        super().__init__(
            rnd_output_dim=rnd_output_dim,
            internal_reward_weight=internal_reward_weight,
            rnd_network_config=rnd_network_config,
            rnd_lr=rnd_lr,
            rnd_eps=rnd_eps,
            rnd_weight_decay=rnd_weight_decay,
            update_proportion=update_proportion,
        )
        self.last_error = 0

    def get_reward(self, metrics):
        if self.rnd_net is None:
            self.initialize_networks(metrics["transition"]["next_state"].shape[1:])

        rnd_error = self.rnd_net.get_error(metrics["transition"]["next_state"])
        metrics["transition"]["intrinsic_reward"] = self.internal_reward_weight * abs(
            rnd_error - self.last_error
        )
        metrics["transition"]["reward"] = metrics["transition"][
            "reward"
        ] + self.internal_reward_weight * abs(rnd_error - self.last_error)
        self.last_error = rnd_error
        return metrics
