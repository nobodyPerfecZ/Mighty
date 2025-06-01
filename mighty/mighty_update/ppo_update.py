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

        policy_params = list(
            model.policy_head.parameters()
        )  # includes feature_extractor_policy
        value_params = list(
            model.value_head.parameters()
        )  # includes feature_extractor_value

        extra_params = []
        if getattr(model, "continuous_action", False) and hasattr(model, "log_std"):
            extra_params.append(model.log_std)

        self.optimizer = optim.Adam(
            [
                {
                    "params": policy_params,
                    "lr": policy_lr,
                },  # policy net (feat-extractor + head)
                {
                    "params": value_params,
                    "lr": value_lr,
                },  # value net  (feat-extractor + head)
                *([{"params": extra_params, "lr": policy_lr}] if extra_params else []),
            ],
            eps=1e-5,
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: 1 - step / float(self.total_steps),
        )

    def update(self, batch: MaxiBatch) -> Dict[str, float]:
        """PPO update with advantage normalisation, epoch-level KL-stop, value-clipping."""

        # cache old values / log-probs with *one* forward pass per minibatch
        with torch.no_grad():
            old_values = [
                self.model.forward_value(mb.observations) for mb in batch.minibatches
            ]
            old_log_probs = [mb.log_probs.clone() for mb in batch.minibatches]

        # global advantage normalisation (once!) and detach from graph
        flat_adv = batch.advantages.squeeze(0)
        adv_mean = flat_adv.mean().detach()
        adv_std = (flat_adv.std() + 1e-8).detach()

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        mb_updates = 0  # mini-batch counter
        epoch_counts = 0  # how many epochs actually executed

        # main PPO loop
        for _ in range(self.n_epochs):
            epoch_kls = []

            for i, mb in enumerate(batch.minibatches):
                # 2a) use the globally normalised advantages
                adv = ((mb.advantages - adv_mean) / adv_std).detach()

                # 2b) value loss (with optional clipping)
                values = self.model.forward_value(mb.observations)
                if self.use_value_clip:
                    clipped = old_values[i] + torch.clamp(
                        values - old_values[i],
                        -self.value_clip_eps,
                        self.value_clip_eps,
                    )
                    v_loss = (
                        0.5
                        * torch.max(
                            (values - mb.returns).pow(2),
                            (clipped - mb.returns).pow(2),
                        ).mean()
                    )
                else:
                    v_loss = 0.5 * (mb.returns - values).pow(2).mean()

                # 2c) new policy log-probs & entropy
                if self.model.continuous_action:
                    # Run forward() to get (action, z, mean, log_std)
                    _, z_pred, mean, log_std = self.model(mb.observations)
                    std = torch.exp(log_std)  # [batch_size, action_dim]
                    dist = torch.distributions.Normal(mean, std)

                    # Compute log_pz = ∑ᵢ log N(z_predᵢ; meanᵢ, stdᵢ)
                    log_pz = dist.log_prob(z_pred).sum(dim=-1)  # [batch_size]

                    # Tanh‐correction: ∑ᵢ log(1 − tanh(z_predᵢ)² + ε)
                    eps = 1e-6
                    log_correction = torch.log(
                        1.0 - torch.tanh(z_pred).pow(2) + eps
                    ).sum(dim=-1)  # [batch_size]

                    # Final logπ(a|s) = log_pz − log_correction
                    log_probs = log_pz - log_correction  # [batch_size]

                    # Entropy = ∑ᵢ H(N(meanᵢ, stdᵢ))
                    entropy = dist.entropy().sum(dim=-1).mean()  # scalar
                else:
                    logits = self.model(mb.observations)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(mb.actions)
                    entropy = dist.entropy().mean()

                # 2d) PPO surrogate loss
                ratios = torch.exp((log_probs - old_log_probs[i]).clamp(-20, 20))
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
                p_loss = -torch.min(surr1, surr2).mean()

                # 2e) total loss, backprop, step
                loss = p_loss + self.vf_coef * v_loss - self.ent_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # 2f) recompute log-probs *after* the step for a correct KL

                with torch.no_grad():
                    if self.model.continuous_action:
                        # After update, forward again
                        _, z_new, mean_new, log_std_new = self.model(mb.observations)
                        std_new = torch.exp(log_std_new)
                        dist_new = torch.distributions.Normal(mean_new, std_new)

                        log_pz_new = dist_new.log_prob(z_new).sum(
                            dim=-1
                        )  # [batch_size]
                        log_correction_new = torch.log(
                            1.0 - torch.tanh(z_new).pow(2) + 1e-6
                        ).sum(dim=-1)  # [batch_size]
                        new_lp = log_pz_new - log_correction_new  # [batch_size]
                    else:
                        logits = self.model(mb.observations)
                        new_lp = torch.distributions.Categorical(
                            logits=logits
                        ).log_prob(mb.actions)
                kl = (old_log_probs[i] - new_lp).mean()
                epoch_kls.append(kl)

                # 2g) log accumulators
                metrics["policy_loss"] += p_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["entropy"] += entropy.item()
                mb_updates += 1

            # epoch finished
            mean_kl = torch.mean(torch.stack(epoch_kls))
            metrics["approx_kl"] += mean_kl.item()
            epoch_counts += 1

            if mean_kl > self.kl_target:
                break  # early-stop epochs

        # LR decay (once per update call)

        self.scheduler.step()

        # 4) final metric averages

        if mb_updates > 0:
            metrics["policy_loss"] /= mb_updates
            metrics["value_loss"] /= mb_updates
            metrics["entropy"] /= mb_updates
        if epoch_counts > 0:
            metrics["approx_kl"] /= epoch_counts

        return {
            "Update/policy_loss": metrics["policy_loss"],
            "Update/value_loss": metrics["value_loss"],
            "Update/entropy": metrics["entropy"],
            "Update/approx_kl": metrics["approx_kl"],
        }
