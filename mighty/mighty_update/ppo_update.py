from typing import Dict

import torch
import torch.optim as optim

from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_replay.mighty_rollout_buffer import MaxiBatch


class PPOUpdate:
    def __init__(
        self,
        model: PPOModel,
        policy_lr: float = 0.001,
        value_lr: float = 0.001,
        epsilon: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        minibatch_size: int = 32,
        kl_target: float = 0.001,
        use_value_clip: bool = True,
        value_clip_eps: float = 0.2,
        total_timesteps: int = 1_000_000,
    ):
        """Initialize PPO update mechanism."""
        self.model = model
        self.epsilon = epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.kl_target = kl_target
        self.use_value_clip = use_value_clip
        self.value_clip_eps = value_clip_eps

        self.total_steps = total_timesteps

        # Optimizers
        policy_params = list(self.model.policy_head.parameters())
        if getattr(self.model, "continuous_action", False) and hasattr(
            self.model, "log_std"
        ):
            policy_params.append(self.model.log_std)
        self.policy_optimizer = optim.Adam(policy_params, lr=policy_lr, eps=1e-5)
        self.value_optimizer = optim.Adam(
            self.model.value_head.parameters(), lr=value_lr
        )

        # Learning rate schedulers (linear decay)
        # FIXME: schedule parameters should come from config
        self.policy_scheduler = optim.lr_scheduler.LambdaLR(
            self.policy_optimizer,
            lr_lambda=lambda step: 1 - step / float(self.total_steps),
        )
        self.value_scheduler = optim.lr_scheduler.LambdaLR(
            self.value_optimizer,
            lr_lambda=lambda step: 1 - step / float(self.total_steps),
        )

    def update(self, batch: MaxiBatch) -> Dict[str, float]:
        """Perform PPO update with multiple epochs, minibatches, KL early stopping, and value clipping."""
        advantages = batch.advantages.squeeze(0)
        adv_mean, adv_std = advantages.mean(), advantages.std() + 1e-8

        # Precompute old values for clipping
        with torch.no_grad():
            old_values = [
                self.model.forward_value(batch.minibatches[i].observations)
                for i in range(len(batch.minibatches))
            ]

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }

        # PPO update epochs
        for _ in range(self.n_epochs):
            for i, mb in enumerate(batch.minibatches):
                # Normalize advantages
                mb.advantages = (mb.advantages - adv_mean) / adv_std

                # Value estimates
                values = self.model.forward_value(mb.observations)

                # Value loss with optional clipping
                if self.use_value_clip:
                    clipped_values = old_values[i] + torch.clamp(
                        values - old_values[i],
                        -self.value_clip_eps,
                        self.value_clip_eps,
                    )

                    loss_unclipped = (values - mb.returns) ** 2
                    loss_clipped = (clipped_values - mb.returns) ** 2
                    value_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
                else:
                    value_loss = 0.5 * (mb.returns - values).pow(2).mean()

                # Compute new log probs & entropy
                if self.model.continuous_action:
                    means, stds = self.model(mb.observations)
                    dist = torch.distributions.Normal(means, stds)
                    log_probs = dist.log_prob(mb.actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    logits = self.model(mb.observations)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(mb.actions)
                    entropy = dist.entropy().mean()

                # Policy objective
                ratios = torch.exp(log_probs - mb.log_probs)
                surr1 = ratios * mb.advantages
                surr2 = (
                    torch.clamp(ratios, 1.0 - self.epsilon, 1.0 + self.epsilon)
                    * mb.advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Approximate KL for early stopping
                approx_kl = (mb.log_probs - log_probs).mean()
                if approx_kl > self.kl_target:
                    break

                # Combined loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Backprop
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()
                self.value_optimizer.step()

                # Accumulate metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["approx_kl"] += approx_kl.item()

            # Scheduler step per epoch
            self.policy_scheduler.step()
            self.value_scheduler.step()

        # Average metrics over updates
        num_updates = self.n_epochs * len(batch.minibatches)
        for k in metrics:
            metrics[k] /= max(1, num_updates)

        return {
            "Update/policy_loss": metrics["policy_loss"],
            "Update/value_loss": metrics["value_loss"],
            "Update/entropy": metrics["entropy"],
            "Update/approx_kl": metrics["approx_kl"],
        }
