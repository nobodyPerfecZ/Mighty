from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from mighty.mighty_models.sac import SACModel
from mighty.mighty_replay.mighty_replay_buffer import TransitionBatch
from mighty.mighty_utils.update_utils import polyak_update


class SACUpdate:
    def __init__(
        self,
        model: SACModel,
        policy_lr: float = 0.001,
        q_lr: float = 0.001,
        tau: float = 0.005,
        alpha: float = 0.2,
        gamma: float = 0.99,
        target_entropy: float = None,
        auto_alpha: bool = True,
        alpha_lr: float = 3e-4,
        policy_frequency: int = 2,
        target_network_frequency: int = 1,
    ):
        self.model = model

        self.policy_optimizer = optim.Adam(model.policy_net.parameters(), lr=policy_lr)
        self.q_optimizer = optim.Adam(
            list(model.q_net1.parameters()) + list(model.q_net2.parameters()), lr=q_lr
        )

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.auto_alpha = auto_alpha
        self.action_dim = self.model.action_size
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.update_step = 0

        if self.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr or q_lr)
            self.target_entropy = (
                -float(self.action_dim)
                if target_entropy is None
                else float(target_entropy)
            )
        else:
            self.alpha = alpha

    def calculate_td_error(self, transition: TransitionBatch) -> Tuple:
        """Calculate the TD error for a given transition."""
        with torch.no_grad():
            a_next, z_next, mean_next, log_std_next = self.model(
                torch.as_tensor(transition.next_obs, dtype=torch.float32)
            )
            logp_next = self.model.policy_log_prob(z_next, mean_next, log_std_next)
            sa_next = torch.cat(
                [
                    torch.as_tensor(transition.next_obs, dtype=torch.float32),
                    a_next,
                ],
                dim=-1,
            )
            q1_t = self.model.target_q_net1(sa_next)
            q2_t = self.model.target_q_net2(sa_next)
            alpha = self.log_alpha.exp() if self.auto_alpha else self.alpha
            q_target = torch.as_tensor(
                transition.rewards, dtype=torch.float32
            ).unsqueeze(-1) + (
                1 - torch.as_tensor(transition.dones, dtype=torch.float32).unsqueeze(-1)
            ) * self.gamma * (torch.min(q1_t, q2_t) - alpha * logp_next)
        sa = torch.cat(
            [
                torch.as_tensor(transition.observations, dtype=torch.float32),
                torch.as_tensor(transition.actions, dtype=torch.float32),
            ],
            dim=-1,
        )
        q1_curr = self.model.q_net1(sa)
        q2_curr = self.model.q_net2(sa)
        td_error1 = q1_curr - q_target
        td_error2 = q2_curr - q_target
        return td_error1, td_error2

    def update(self, batch: TransitionBatch) -> Dict:
        """Perform an update of the SAC model using a batch of experience."""
        self.update_step += 1

        states = torch.as_tensor(batch.observations, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.float32)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).unsqueeze(-1)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32).unsqueeze(-1)
        next_states = torch.as_tensor(batch.next_obs, dtype=torch.float32)

        # --- Q-network update ---
        with torch.no_grad():
            a_next, z_next, mean_next, log_std_next = self.model(next_states)
            logp_next = self.model.policy_log_prob(z_next, mean_next, log_std_next)
            sa_next = torch.cat([next_states, a_next], dim=-1)
            q1_t = self.model.target_q_net1(sa_next)
            q2_t = self.model.target_q_net2(sa_next)
            current_alpha = (
                self.log_alpha.exp().detach() if self.auto_alpha else self.alpha
            )
            q_target = rewards + (1 - dones) * self.gamma * (
                torch.min(q1_t, q2_t) - current_alpha * logp_next
            )

        sa = torch.cat([states, actions], dim=-1)
        q1 = self.model.q_net1(sa)
        q2 = self.model.q_net2(sa)
        q_loss1 = F.mse_loss(q1, q_target)
        q_loss2 = F.mse_loss(q2, q_target)
        q_loss = q_loss1 + q_loss2

        # use combined optimizer for both Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy update (delayed) ---
        policy_loss = torch.tensor(0.0)
        alpha_loss = torch.tensor(0.0)
        if self.update_step % self.policy_frequency == 0:
            # do multiple policy updates to compensate for delay
            for _ in range(self.policy_frequency):
                # recompute alpha after q update
                current_alpha = (
                    self.log_alpha.exp().detach() if self.auto_alpha else self.alpha
                )

                # Sample fresh actions for each policy update iteration
                # This ensures stochasticity across iterations
                a, z, mean, log_std = self.model(states)
                logp = self.model.policy_log_prob(z, mean, log_std)
                sa_pi = torch.cat([states, a], dim=-1)
                
                q1_pi = self.model.q_net1(sa_pi)
                q2_pi = self.model.q_net2(sa_pi)
                q_pi = torch.min(q1_pi, q2_pi)
                policy_loss = (current_alpha * logp - q_pi).mean()

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                # --- Entropy coefficient (alpha) update ---
                if self.auto_alpha:
                    # Get fresh sample for alpha update
                    with torch.no_grad():
                        _, z_alpha, mean_alpha, log_std_alpha = self.model(states)
                        logp_alpha = self.model.policy_log_prob(z_alpha, mean_alpha, log_std_alpha)
                    
                    alpha_loss = -(
                        self.log_alpha.exp() * (logp_alpha.detach() + self.target_entropy)
                    ).mean()
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # --- Soft update targets ---
        if self.update_step % self.target_network_frequency == 0:
            polyak_update(
                self.model.q_net1.parameters(),
                self.model.target_q_net1.parameters(),
                self.tau,
            )
            polyak_update(
                self.model.q_net2.parameters(),
                self.model.target_q_net2.parameters(),
                self.tau,
            )

        # --- Logging metrics ---
        td1, td2 = self.calculate_td_error(batch)
        return {
            "Update/q_loss1": q_loss1.item(),
            "Update/q_loss2": q_loss2.item(),
            "Update/policy_loss": policy_loss.item(),
            "Update/alpha_loss": alpha_loss.item(),
            "Update/td_error1": td1.mean().item(),
            "Update/td_error2": td2.mean().item(),
        }
