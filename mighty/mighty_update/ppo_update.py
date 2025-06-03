from typing import Dict

import torch
import torch.optim as optim

from mighty.mighty_models.ppo import PPOModel
from mighty.mighty_replay.mighty_rollout_buffer import MaxiBatch


class PPOUpdate:
    def __init__(
        self,
        model: PPOModel,
        policy_lr: float = 3e-4,  # INCREASED from 1e-8 to reasonable value
        value_lr: float = 3e-4,  # INCREASED from 1e-8 to reasonable value
        epsilon: float = 0.1,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        minibatch_size: int = 32,
        kl_target: float = 0.01,
        use_value_clip: bool = True,
        value_clip_eps: float = 0.2,
        total_timesteps: int = 1_000_000,
        adaptive_lr: bool = True,  # Enable adaptive LR management
        min_lr: float = 1e-6,  # Minimum LR threshold
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
        self.adaptive_lr = adaptive_lr
        self.min_lr = min_lr

        self.total_steps = total_timesteps

        # Store initial learning rates
        self.initial_policy_lr = policy_lr
        self.initial_value_lr = value_lr

        # Optimizers
        policy_params = list(model.policy_head.parameters())
        value_params = list(model.value_head.parameters())

        extra_params = []
        if getattr(model, "continuous_action", False) and hasattr(model, "log_std"):
            extra_params.append(model.log_std)

        print(f"[DEBUG] Initial Policy LR: {policy_lr:.2e}, Value LR: {value_lr:.2e}")
        print(f"[DEBUG] PPO clip ε = {self.epsilon}")
        print(f"[DEBUG] KL target = {self.kl_target}")

        self.optimizer = optim.Adam(
            [
                {"params": policy_params, "lr": policy_lr},
                {"params": value_params, "lr": value_lr},
                *([{"params": extra_params, "lr": policy_lr}] if extra_params else []),
            ],
            eps=1e-5,
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(
                0.1, 1 - step / float(self.total_steps)
            ),  # Don't go below 10% of initial
        )

    def update(self, batch: MaxiBatch) -> Dict[str, float]:
        """PPO update with corrected KL calculation and adaptive LR."""

        # Cache old values/log-probs
        with torch.no_grad():
            old_values = [
                self.model.forward_value(mb.observations) for mb in batch.minibatches
            ]
            old_log_probs = [mb.log_probs.clone() for mb in batch.minibatches]

            # Store original actions for consistent KL calculation
            original_actions = [mb.actions.clone() for mb in batch.minibatches]

        # Global advantage normalization
        flat_adv = batch.advantages.squeeze(0)
        adv_mean = flat_adv.mean().detach()
        adv_std = (flat_adv.std() + 1e-8).detach()

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
        }
        mb_updates = 0
        epoch_counts = 0

        # Main PPO loop
        for epoch in range(self.n_epochs):
            epoch_kls = []

            for i, mb in enumerate(batch.minibatches):
                # Normalized advantages
                adv = ((mb.advantages - adv_mean) / adv_std).detach()

                # Value loss
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

                # Policy forward pass
                if self.model.continuous_action:
                    _, z_pred, mean, log_std = self.model(mb.observations)
                    std = torch.exp(log_std)
                    dist = torch.distributions.Normal(mean, std)

                    log_pz = dist.log_prob(z_pred).sum(dim=-1)
                    eps = 1e-6
                    log_correction = torch.log(
                        1.0 - torch.tanh(z_pred).pow(2) + eps
                    ).sum(dim=-1)
                    log_probs = log_pz - log_correction
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    logits = self.model(mb.observations)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_probs = dist.log_prob(mb.actions)
                    entropy = dist.entropy().mean()

                # PPO surrogate loss
                raw_score = log_probs - old_log_probs[i]
                ratios = torch.exp(raw_score)
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv
                p_loss = -torch.min(surr1, surr2).mean()

                # Total loss and optimization step
                loss = p_loss + self.vf_coef * v_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # KL calculation using original actions (FIXED)
                with torch.no_grad():
                    if self.model.continuous_action:
                        # Get updated policy parameters
                        _, _, mean_new, log_std_new = self.model(mb.observations)
                        std_new = torch.exp(log_std_new)
                        dist_new = torch.distributions.Normal(mean_new, std_new)

                        # Use ORIGINAL actions for consistent KL
                        orig_actions = original_actions[i]
                        log_pz_new = dist_new.log_prob(orig_actions).sum(dim=-1)
                        log_correction_new = torch.log(
                            1.0 - torch.tanh(orig_actions).pow(2) + 1e-6
                        ).sum(dim=-1)
                        new_lp = log_pz_new - log_correction_new
                    else:
                        logits_new = self.model(mb.observations)
                        new_lp = torch.distributions.Categorical(
                            logits=logits_new
                        ).log_prob(mb.actions)

                kl = (old_log_probs[i] - new_lp).mean()
                epoch_kls.append(kl)

                # Accumulate metrics
                metrics["policy_loss"] += p_loss.item()
                metrics["value_loss"] += v_loss.item()
                metrics["entropy"] += entropy.item()
                mb_updates += 1

            # Epoch finished
            mean_kl = torch.mean(torch.stack(epoch_kls))
            epoch_counts += 1

            # print(f"[PPO] Epoch {epoch}: KL={mean_kl:.6f}, target={self.kl_target:.6f}")

            # Adaptive learning rate management
            if self.adaptive_lr:
                current_policy_lr = self.optimizer.param_groups[0]["lr"]
                current_value_lr = self.optimizer.param_groups[1]["lr"]

                if mean_kl > 1.5 * self.kl_target:
                    # KL too high - reduce LR
                    new_policy_lr = max(current_policy_lr * 0.8, self.min_lr)
                    new_value_lr = max(current_value_lr * 0.8, self.min_lr)
                    self.optimizer.param_groups[0]["lr"] = new_policy_lr
                    self.optimizer.param_groups[1]["lr"] = new_value_lr
                    # print(f"[ADAPTIVE] Reduced LR: policy={new_policy_lr:.2e}, value={new_value_lr:.2e}")

                elif mean_kl < 0.5 * self.kl_target and epoch == 0:
                    # KL too low - can increase LR slightly
                    new_policy_lr = min(current_policy_lr * 1.1, self.initial_policy_lr)
                    new_value_lr = min(current_value_lr * 1.1, self.initial_value_lr)
                    self.optimizer.param_groups[0]["lr"] = new_policy_lr
                    self.optimizer.param_groups[1]["lr"] = new_value_lr
                    # print(f"[ADAPTIVE] Increased LR: policy={new_policy_lr:.2e}, value={new_value_lr:.2e}")

            # Store KL for reporting (only once per update)
            if epoch == 0:
                metrics["approx_kl"] = mean_kl.item()

            # Early stopping
            if mean_kl > self.kl_target:
                # print(f"[EARLY STOP] KL {mean_kl:.6f} > target {self.kl_target:.6f} at epoch {epoch}")
                break

        # LR scheduler step (but don't let it go too low)
        # old_lrs = [group["lr"] for group in self.optimizer.param_groups]
        self.scheduler.step()
        # new_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Enforce minimum learning rate
        for i, group in enumerate(self.optimizer.param_groups):
            if group["lr"] < self.min_lr:
                group["lr"] = self.min_lr
                print(f"[LR] Enforced minimum LR {self.min_lr:.2e} for param group {i}")

        # Final metrics
        if mb_updates > 0:
            metrics["policy_loss"] /= mb_updates
            metrics["value_loss"] /= mb_updates
            metrics["entropy"] /= mb_updates

        return {
            "Update/policy_loss": metrics["policy_loss"],
            "Update/value_loss": metrics["value_loss"],
            "Update/entropy": metrics["entropy"],
            "Update/approx_kl": metrics["approx_kl"],
        }
