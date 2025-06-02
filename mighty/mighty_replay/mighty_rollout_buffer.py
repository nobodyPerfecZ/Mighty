# mighty_rollout_buffer.py

from __future__ import annotations

import dill as pickle
import numpy as np
import torch

from mighty.mighty_replay.buffer import MightyBuffer


class MaxiBatch:
    def __init__(self, minibatches: list[RolloutBatch]):
        self.minibatches = minibatches

    @property
    def size(self) -> int:
        return sum(len(mb.observations) for mb in self.minibatches)

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from self.minibatches

    def __getattribute__(self, name: str):
        if name in [
            "observations",
            "actions",
            "rewards",
            "advantages",
            "returns",
            "episode_starts",
            "log_probs",
            "values",
        ]:
            minibatches = object.__getattribute__(self, "minibatches")
            if not minibatches:
                return torch.tensor([])

            stacked = torch.stack([getattr(mb, name) for mb in minibatches], dim=0)
            feat_shape = getattr(minibatches[0], name).shape[1:]
            return stacked.reshape(-1, *feat_shape)
        else:
            return object.__getattribute__(self, name)


class RolloutBatch:
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        episode_starts: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        device: torch.device | str = "cpu",
    ):
        self.device = device

        obs_t = torch.from_numpy(observations.astype(np.float32))
        act_t = torch.from_numpy(actions.astype(np.float32))
        rew_t = torch.from_numpy(rewards.astype(np.float32))
        adv_t = torch.from_numpy(advantages.astype(np.float32))
        ret_t = torch.from_numpy(returns.astype(np.float32))
        eps_t = torch.from_numpy(episode_starts.astype(np.float32))
        logp_t = torch.from_numpy(log_probs.astype(np.float32))
        val_t = torch.from_numpy(values.astype(np.float32))

        # Promote obs from [n_envs, obs_dim] → [1, n_envs, obs_dim] if needed
        if obs_t.dim() == 2:
            obs_t = obs_t.unsqueeze(0)
        elif obs_t.dim() < 2:
            raise RuntimeError(
                f"RolloutBatch: `observations` must be ≥2-D, got {obs_t.shape}"
            )

        def _promote_to_2d(x: torch.Tensor, name: str):
            if x.dim() == 1:
                return x.unsqueeze(0)
            elif x.dim() == 2:
                return x
            else:
                raise RuntimeError(
                    f"RolloutBatch: `{name}` must be 1-D or 2-D, got {x.shape}"
                )

        act_t = _promote_to_2d(act_t, "actions")
        rew_t = _promote_to_2d(rew_t, "rewards")
        adv_t = _promote_to_2d(adv_t, "advantages")
        ret_t = _promote_to_2d(ret_t, "returns")
        eps_t = _promote_to_2d(eps_t, "episode_starts")
        logp_t = _promote_to_2d(logp_t, "log_probs")
        val_t = _promote_to_2d(val_t, "values")

        self.observations = obs_t.to(self.device)
        self.actions = act_t.to(self.device)
        self.rewards = rew_t.to(self.device)
        self.advantages = adv_t.to(self.device)
        self.returns = ret_t.to(self.device)
        self.episode_starts = eps_t.to(self.device)
        self.log_probs = logp_t.to(self.device)
        self.values = val_t.to(self.device)

    @property
    def size(self) -> int:
        return self.observations.shape[0]

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        yield from zip(
            self.observations,
            self.actions,
            self.rewards,
            self.advantages,
            self.returns,
            self.episode_starts,
            self.log_probs,
            self.values,
            strict=False,
        )


class MightyRolloutBuffer(MightyBuffer):
    """
    Pre-allocated rollout buffer (no repeated concat).
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        act_dim,
        device: torch.device | str = "cpu",
        gae_lambda: float = 1.0,
        gamma: float = 0.99,
        n_envs: int = 1,
        discrete_action: bool = False,
    ):
        super().__init__()
        self.buffer_size = buffer_size  # maximum number of time-steps
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.discrete_action = discrete_action

        # If obs_shape is an integer, convert to tuple
        if isinstance(obs_shape, int):
            obs_shape = (obs_shape,)
        elif isinstance(obs_shape, tuple):
            # nothing to do
            pass
        else:
            raise RuntimeError(
                f"obs_shape must be int or tuple[int,...], got {obs_shape!r}"
            )

        # print(f'Buffer shape: {buffer_size}, n_envs: {n_envs}, obs_shape: {obs_shape}, act_dim: {act_dim}, device: {device}')

        # — Pre-allocate everything once on `device` —

        self.observations = torch.zeros(
            (buffer_size, n_envs, *obs_shape),
            dtype=torch.float32,
            device=device,  # shape = [buffer_size, n_envs, *obs_shape]
        )

        # actions might be scalar (e.g. discrete) or multi-dim; here we store as float32
        if self.discrete_action:
            # Discrete: a single integer‐per‐env per timestep → shape [buffer_size, n_envs]
            self.actions = torch.zeros(
                (buffer_size, n_envs),
                dtype=torch.float32,  # still float32 so we can cast / compare later
                device=device,
            )
        else:
            # Continuous: an `act_dim`-vector per env per timestep → shape [buffer_size, n_envs, act_dim]
            self.actions = torch.zeros(
                (buffer_size, n_envs, act_dim),
                dtype=torch.float32,
                device=device,
            )

        self.rewards = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )
        self.advantages = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )
        self.returns = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )
        self.episode_starts = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )
        self.log_probs = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )
        self.values = torch.zeros(
            (buffer_size, n_envs), dtype=torch.float32, device=device
        )

        # `pos` tells us how many time-steps have been filled so far.
        self.pos = 0
        print("Buffer capacity =", self.observations.shape[0])

    def reset(self) -> None:
        # Just zero out pos; no need to re-allocate
        self.pos = 0

    # def compute_returns_and_advantage(
    #     self,
    #     last_values: torch.Tensor,  # shape=[n_envs] or [n_envs,1]
    #     dones: np.ndarray,          # shape=[n_envs]
    # ) -> None:
    #     if self.pos == 0:
    #         return

    #     # 1) last_values → 1-D tensor [n_envs]
    #     lv = last_values.clone().to(self.device).reshape(-1)
    #     # 2) dones → float tensor [n_envs]
    #     dones_t = (
    #         torch.from_numpy(dones.astype(np.float32))
    #         .to(self.device)
    #         .reshape(-1)
    #     )

    #     T = self.pos    # number of filled time-steps
    #     N = self.n_envs

    #     # Use only the slice [0:T] for each buffer field
    #     rew_slice = self.rewards[:T]            # shape=[T, n_envs]
    #     val_slice = self.values[:T]              # shape=[T, n_envs]
    #     eps_slice = self.episode_starts[:T]      # shape=[T, n_envs]
    #     adv_slice = self.advantages[:T]         # shape=[T, n_envs]

    #     last_gae = torch.zeros(N, device=self.device)

    #     for step in reversed(range(T)):
    #         if step == T - 1:
    #             next_non_term = 1.0 - dones_t     # [n_envs]
    #             next_val = lv                     # [n_envs]
    #         else:
    #             next_non_term = 1.0 - eps_slice[step + 1]
    #             next_val = val_slice[step + 1]

    #         r_t = rew_slice[step]
    #         v_t = val_slice[step]

    #         delta = r_t + self.gamma * next_val * next_non_term - v_t
    #         last_gae = delta + self.gamma * self.gae_lambda * next_non_term * last_gae
    #         adv_slice[step] = last_gae

    #     # Write advantages and returns back into the buffer
    #     self.advantages[:T] = adv_slice
    #     self.returns[:T] = adv_slice + val_slice

    #     # with torch.no_grad():
    #     #     ret_min, ret_max = self.returns.min().item(), self.returns.max().item()
    #     #     adv_min, adv_max = self.advantages.min().item(), self.advantages.max().item()
    #     #     print(f"[DEBUG - RETURN/ADV] returns ∈ [{ret_min:.1f}, {ret_max:.1f}];  adv ∈ [{adv_min:.3f}, {adv_max:.3f}]")

    def compute_returns_and_advantage(
        self,
        last_values: torch.Tensor,  # shape = [n_envs] or [n_envs, 1]
        dones: np.ndarray,  # shape = [n_envs]
    ) -> None:
        if self.pos == 0:
            return

        # 1) Turn last_values into a 1‐D tensor of shape [n_envs]
        lv = last_values.clone().to(self.device).reshape(-1)  # → [n_envs]

        # 2) Turn the numpy dones (0/1) into float tensor on device, shape [n_envs]
        dones_t = (
            torch.from_numpy(dones.astype(np.float32)).to(self.device).reshape(-1)
        )  # → [n_envs]

        T = self.pos  # number of filled “time‐steps” in buffer
        N = self.n_envs  # number of parallel envs

        # 3) Slice out exactly the first T entries along each field
        #    Each of these has shape [T, n_envs]
        rew_slice = self.rewards[:T]  # (T × n_envs)
        val_slice = self.values[:T]  # (T × n_envs)
        eps_slice = self.episode_starts[:T]  # (T × n_envs)  ← “episode_starts” flags
        adv_slice = self.advantages[:T]  # (T × n_envs), but usually zero‐initialized

        # 4) Initialize the last‐step GAE “accumulator” to zero for each env
        last_gae = torch.zeros(N, device=self.device)  # [n_envs]

        # 5) Walk backwards over time steps
        for step in reversed(range(T)):
            if step == T - 1:
                # On the very last (most recent) buffer row:
                #   • If done[i] == 1, that env actually terminated (pole fell)
                #   • If done[i] == 0, that env was truncated (hit 500) or still “alive”
                next_non_term = 1.0 - dones_t  # [n_envs], 0 if done, 1 otherwise
                next_val = lv  # bootstrap from V(sₜ₊₁)
            else:
                # On intermediate steps, look at “episode_starts” for whether step+1 was a new episode
                next_non_term = 1.0 - eps_slice[step + 1]  # [n_envs]
                next_val = val_slice[step + 1]  # [n_envs]

            r_t = rew_slice[step]  # shape = [n_envs]
            v_t = val_slice[step]  # shape = [n_envs]

            # standard TD residual
            delta = r_t + self.gamma * next_val * next_non_term - v_t  # [n_envs]

            # recursive GAE
            last_gae = (
                delta + self.gamma * self.gae_lambda * next_non_term * last_gae
            )  # [n_envs]

            # store into advantage‐buffer
            adv_slice[step] = last_gae

        # 6) write the final advantages & returns back into the buffer
        self.advantages[:T] = adv_slice
        self.returns[:T] = adv_slice + val_slice

        # with torch.no_grad():
        #     ret_min, ret_max = self.returns[:T].min().item(), self.returns[:T].max().item()
        #     adv_min, adv_max = self.advantages[:T].min().item(), self.advantages[:T].max().item()
        #     print(f"[GAE] returns ∈ [{ret_min:.1f}, {ret_max:.1f}]; adv ∈ [{adv_min:.3f}, {adv_max:.3f}]")

    def add(self, rollout_batch: RolloutBatch, _=None):
        """
        In-place write into a pre-allocated tensor. Does 2-D→3-D promotion if needed.
        """
        # pull new data onto device
        rb_obs = rollout_batch.observations.to(
            self.device
        )  # [n_steps, n_envs, *obs_shape]
        rb_acts = rollout_batch.actions.to(self.device)  # [n_steps, n_envs]
        rb_rews = rollout_batch.rewards.to(self.device)  # [n_steps, n_envs]
        rb_advs = rollout_batch.advantages.to(self.device)  # [n_steps, n_envs]
        rb_rets = rollout_batch.returns.to(self.device)  # [n_steps, n_envs]
        rb_eps = rollout_batch.episode_starts.to(self.device)  # [n_steps, n_envs]
        rb_logp = rollout_batch.log_probs.to(self.device)  # [n_steps, n_envs]
        rb_vals = rollout_batch.values.to(self.device)  # [n_steps, n_envs]

        # Promote dims if necessary (same logic as before)
        if rb_obs.dim() == 2:
            rb_obs = rb_obs.unsqueeze(0)
        if rb_acts.dim() == 1:
            rb_acts = rb_acts.unsqueeze(0)
        if rb_rews.dim() == 1:
            rb_rews = rb_rews.unsqueeze(0)
        if rb_advs.dim() == 1:
            rb_advs = rb_advs.unsqueeze(0)
        if rb_rets.dim() == 1:
            rb_rets = rb_rets.unsqueeze(0)
        if rb_eps.dim() == 1:
            rb_eps = rb_eps.unsqueeze(0)
        if rb_logp.dim() == 1:
            rb_logp = rb_logp.unsqueeze(0)
        if rb_vals.dim() == 1:
            rb_vals = rb_vals.unsqueeze(0)

        n_steps = rb_obs.shape[0]  # usually 1, but could be >1
        T = self.buffer_size

        if self.pos + n_steps > T:
            raise RuntimeError(
                f"Buffer overflow: pos={self.pos}, trying to add {n_steps} steps but buffer_size={T}"
            )

        # In-place copy into the pre-allocated tensor
        self.observations[self.pos : self.pos + n_steps] = rb_obs
        self.actions[self.pos : self.pos + n_steps] = rb_acts
        self.rewards[self.pos : self.pos + n_steps] = rb_rews
        self.advantages[self.pos : self.pos + n_steps] = rb_advs
        self.returns[self.pos : self.pos + n_steps] = rb_rets
        self.episode_starts[self.pos : self.pos + n_steps] = rb_eps
        self.log_probs[self.pos : self.pos + n_steps] = rb_logp.T  # FIXME: Hack for now
        self.values[self.pos : self.pos + n_steps] = rb_vals

        self.pos += n_steps

    def sample(self, batch_size: int) -> MaxiBatch:
        """
        1) Flatten only the filled portion [0:self.pos] of each pre-allocated tensor.
        2) Shuffle & slice into minibatches of size `batch_size`.
        """
        if self.pos == 0:
            return MaxiBatch([])

        T = self.pos  # actual filled length
        # N = self.n_envs

        # Flatten [T, n_envs, ...] → [T*N, ...], or [T, n_envs] → [T*N]
        def _flatten(t: torch.Tensor) -> torch.Tensor:
            # First slice to only the filled portion
            t_slice = t[:T]
            shape = t_slice.shape
            # If t_slice has ≥2 dims, collapse first two:
            if len(shape) >= 2:
                return t_slice.reshape(shape[0] * shape[1], *shape[2:])
            else:
                return t_slice.reshape(-1)

        # Now call _flatten on each field (after moving to CPU)
        obs_flat = _flatten(self.observations.cpu())
        acts_flat = _flatten(self.actions.cpu())
        rews_flat = _flatten(self.rewards.cpu())
        advs_flat = _flatten(self.advantages.cpu())
        rets_flat = _flatten(self.returns.cpu())
        eps_flat = _flatten(self.episode_starts.cpu())
        logps_flat = _flatten(self.log_probs.cpu())
        vals_flat = _flatten(self.values.cpu())

        # Check that every flattened tensor has the same length
        lengths = {
            "obs": obs_flat.shape[0],
            "acts": acts_flat.shape[0],
            "rews": rews_flat.shape[0],
            "advs": advs_flat.shape[0],
            "rets": rets_flat.shape[0],
            "eps": eps_flat.shape[0],
            "logps": logps_flat.shape[0],
            "vals": vals_flat.shape[0],
        }
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            print("=== Buffer shapes before flatten ===")
            print(f"  observations:   {self.observations[:T].shape}")
            print(f"  actions:        {self.actions[:T].shape}")
            print(f"  rewards:        {self.rewards[:T].shape}")
            print(f"  advantages:     {self.advantages[:T].shape}")
            print(f"  returns:        {self.returns[:T].shape}")
            print(f"  episode_starts: {self.episode_starts[:T].shape}")
            print(f"  log_probs:      {self.log_probs[:T].shape}")
            print(f"  values:         {self.values[:T].shape}")
            print("→ After flatten, lengths were:", lengths)
            raise RuntimeError(
                "Buffer fields have mismatched flattened lengths: "
                + ", ".join(f"{k}={v}" for k, v in lengths.items())
            )

        total = unique_lengths.pop()
        if total < batch_size:
            return MaxiBatch([])

        perm = np.random.permutation(total)
        n_full = total // batch_size
        perm = perm[: n_full * batch_size]
        perm = perm.reshape(n_full, batch_size)

        mini_batches: list[RolloutBatch] = []
        for inds in perm:
            obs_b = obs_flat[inds].numpy()
            acts_b = acts_flat[inds].numpy()
            rews_b = rews_flat[inds].numpy()
            advs_b = advs_flat[inds].numpy()
            rets_b = rets_flat[inds].numpy()
            eps_b = eps_flat[inds].numpy()
            logps_b = logps_flat[inds].numpy()
            vals_b = vals_flat[inds].numpy()

            mini_batches.append(
                RolloutBatch(
                    observations=obs_b,
                    actions=acts_b,
                    rewards=rews_b,
                    advantages=advs_b,
                    returns=rets_b,
                    episode_starts=eps_b,
                    log_probs=logps_b,
                    values=vals_b,
                )
            )

        return MaxiBatch(mini_batches)

    def __len__(self) -> int:
        return self.pos * self.n_envs

    def __bool__(self) -> bool:
        return self.pos > 0

    def save(self, filename="buffer.pkl") -> None:
        with open(filename, "wb") as f:
            pickle.dump(self, f)
