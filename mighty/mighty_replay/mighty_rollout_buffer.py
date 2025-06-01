"""Mighty rollout buffer."""

from __future__ import annotations

import dill as pickle
import numpy as np
import torch

from mighty.mighty_replay.buffer import MightyBuffer


class MaxiBatch:
    def __init__(self, minibatches):
        """
        Initialize a batch of rollout transitions.

        :param minibatches: List of MiniBatch objects.
        """
        self.minibatches = minibatches

    @property
    def size(self):
        """
        Return the number of transitions in the batch.

        :return: Number of transitions.
        """
        return sum(
            [
                len(self.minibatches[i].observations)
                for i in range(len(self.minibatches))
            ]
        )

    def __len__(self):
        """
        Return the number of transitions in the batch.

        :return: Number of transitions.
        """
        return self.size

    def __iter__(self):
        """
        Iterate over the minibatches in the batch.

        :yield: Tuples of (observation, action, reward, advantage, return, episode_start, log_prob, value).
        """
        yield from self.minibatches

    def __getattribute__(self, name):
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
            batch_stack = torch.stack([getattr(mb, name) for mb in self.minibatches])
            batch_stack = batch_stack.reshape(
                (-1, *getattr(self.minibatches[0], name).squeeze().shape[1:])
            )
            return batch_stack
        else:
            return object.__getattribute__(self, name)


class RolloutBatch:
    def __init__(
        self,
        observations,
        actions,
        rewards,
        advantages,
        returns,
        episode_starts,
        log_probs,
        values,
        device: torch.device | str = "cpu",
    ):
        """
        Initialize a batch of rollout transitions, immediately moving everything to `device`.

        :param observations: NumPy array with shape (n_steps, n_envs, *obs_shape)
        :param actions:      NumPy array with shape (n_steps, n_envs, *action_shape)
        :param rewards:      NumPy array with shape (n_steps, n_envs)
        :param advantages:   NumPy array with shape (n_steps, n_envs)
        :param returns:      NumPy array with shape (n_steps, n_envs)
        :param episode_starts: NumPy array with shape (n_steps, n_envs)
        :param log_probs:    NumPy array with shape (n_steps, n_envs)
        :param values:       NumPy array with shape (n_steps, n_envs)
        :param device:       Torch device (e.g. "cpu", "cuda:0")
        """
        self.device = device

        self.observations = (
            torch.from_numpy(observations.astype(np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        self.actions = (
            torch.from_numpy(actions.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        self.rewards = (
            torch.from_numpy(rewards.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        self.advantages = (
            torch.from_numpy(advantages.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        self.returns = (
            torch.from_numpy(returns.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        self.episode_starts = (
            torch.from_numpy(episode_starts.astype(np.float32))
            .unsqueeze(0)
            .to(self.device)
        )
        self.log_probs = (
            torch.from_numpy(log_probs.astype(np.float32)).unsqueeze(0).to(self.device)
        )
        self.values = (
            torch.from_numpy(values.astype(np.float32)).unsqueeze(0).to(self.device)
        )

    @property
    def size(self):
        """
        Return the number of transitions in the batch.

        :return: Number of transitions.
        """
        return len(self.observations)

    def __len__(self):
        """
        Return the number of transitions in the batch.

        :return: Number of transitions.
        """
        return self.size

    def __iter__(self):
        """
        Iterate over the transitions in the batch.

        :yield: Tuples of (observation, action, reward, advantage, return, episode_start, log_prob, value).
        """
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
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    Stores transitions and computes returns and advantages.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_shape,
        act_dim,
        device: torch.device | str = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        """
        Initialize the rollout buffer.

        :param buffer_size: Maximum number of transitions to store.
        :param obs_shape: Shape of the observation space.
        :param act_dim: Dimension of the action space.
        :param device: Device to store tensors on.
        :param gae_lambda: Lambda parameter for GAE.
        :param gamma: Discount factor.
        :param n_envs: Number of parallel environments.
        """
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.act_dim = act_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.reset()

    def reset(self) -> None:
        """
        Reset the buffer by clearing all stored transitions.
        """
        self.observations = []
        self.actions = []
        self.rewards = []
        self.returns = []
        self.episode_starts = []
        self.values = []
        self.log_probs = []
        self.advantages = []
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(
        self, last_values: torch.Tensor, dones: np.ndarray
    ) -> None:
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).

        :param last_values: Value estimates for the last observation of each environment (shape: [n_envs]).
        :param dones: Done flags for the last step of each environment (shape: [n_envs]).
        """

        last_values = last_values.clone().cpu().squeeze(1)  # [n_envs]
        last_gae_lam = 0  # [n_envs], will be broadcasted as needed

        for step in reversed(
            range(self.observations.shape[0])
        ):  # step: int, loop over [num_steps]
            if step == self.observations.shape[0] - 1:
                # For the last step, use the dones and last_values provided
                next_non_terminal = torch.FloatTensor(
                    1.0 - dones.astype(np.float32)
                )  # [n_envs]
                next_values = last_values  # [n_envs]
            else:
                # For other steps, use episode_starts to determine if next state is terminal
                next_non_terminal = 1.0 - self.episode_starts[step + 1]  # [n_envs]
                next_values = self.values[step + 1]  # [n_envs]

            # Compute the TD residual (delta) for GAE
            # self.rewards[step]: [n_envs]
            # next_values: [n_envs]
            # next_non_terminal: [n_envs]
            # self.values[step]: [n_envs]
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )  # [n_envs]

            # Recursive GAE computation
            # last_gae_lam: [n_envs]
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )  # [n_envs]

            # Store the computed advantage for this step
            self.advantages[step] = last_gae_lam  # [n_envs]

        # Compute returns as sum of advantages and values
        # self.advantages: [num_steps, n_envs], self.values: [num_steps, n_envs]
        self.returns = self.advantages + self.values  # [num_steps, n_envs]

    def add(self, rollout_batch: RolloutBatch, _):
        """
        Add a batch of transitions to the buffer.

        :param rollout_batch: RolloutBatch containing transitions to add.
        :param _: Unused argument (for compatibility).
        """

        # Move each field in rollout_batch onto self.device (if it isn’t already).
        rb_obs = rollout_batch.observations.to(self.device)
        rb_acts = rollout_batch.actions.to(self.device)
        rb_rews = rollout_batch.rewards.to(self.device)
        rb_advs = rollout_batch.advantages.to(self.device)
        rb_rets = rollout_batch.returns.to(self.device)
        rb_eps = rollout_batch.episode_starts.to(self.device)
        rb_logp = rollout_batch.log_probs.to(self.device)
        rb_vals = rollout_batch.values.to(self.device)

        # If buffer was empty (first insertion), just assign:
        if not isinstance(self.observations, torch.Tensor):
            self.observations = rb_obs  # shape = (1, T, n_envs, *obs_shape)
            self.actions = rb_acts  # shape = (1, T, n_envs, *act_shape)
            self.rewards = rb_rews  # shape = (1, T, n_envs)
            self.advantages = rb_advs  # shape = (1, T, n_envs)
            self.returns = rb_rets  # shape = (1, T, n_envs)
            self.episode_starts = rb_eps  # shape = (1, T, n_envs)
            self.log_probs = rb_logp  # shape = (1, T, n_envs)
            self.values = rb_vals  # shape = (1, T, n_envs)

        else:
            # Concatenate along dim=1 (time dimension)
            self.observations = torch.cat((self.observations, rb_obs), dim=1)
            self.actions = torch.cat((self.actions, rb_acts), dim=1)
            self.rewards = torch.cat((self.rewards, rb_rews), dim=1)
            self.advantages = torch.cat((self.advantages, rb_advs), dim=1)
            self.returns = torch.cat((self.returns, rb_rets), dim=1)
            self.episode_starts = torch.cat((self.episode_starts, rb_eps), dim=1)
            self.log_probs = torch.cat((self.log_probs, rb_logp), dim=1)
            self.values = torch.cat((self.values, rb_vals), dim=1)

    # def sample(self, batch_size: int):
    #     """
    #     Sample mini-batches of transitions from the buffer.

    #     :param batch_size: Number of transitions per mini-batch.
    #     :return: List of RolloutBatch samples.
    #     """
    #
    #     hangover = len(self.observations) % batch_size
    #     # import pdb; pdb.set_trace()
    #     indices = np.random.permutation(len(self.observations))
    #     indices = (
    #         indices[:-hangover].reshape(-1, batch_size).tolist()
    #         + indices[-hangover:].tolist()
    #     )
    #     samples = []
    #     for ind in indices:
    #         samples.append(self._get_samples(ind))

    #     return MaxiBatch(samples)

    # def sample(self, batch_size: int):
    #     """
    #     Sample mini‐batches of exactly `batch_size` transitions,
    #     dropping any leftover indices so all batches have equal size.
    #     """
    #     total = len(self.observations)  # total # of stored transitions
    #     # randomly permute indices
    #     perm = np.random.permutation(total)
    #     # compute how many full batches we can make
    #     n_full = total // batch_size
    #     if n_full == 0:
    #         # not enough samples for even one full batch
    #         return MaxiBatch([])

    #     # keep only the first n_full * batch_size indices
    #     perm = perm[: n_full * batch_size]
    #     # reshape into (n_full, batch_size)
    #     perm = perm.reshape(n_full, batch_size)

    #     samples = []
    #     for batch_inds in perm:
    #         samples.append(self._get_samples(batch_inds))

    #     import pdb; pdb.set_trace()

    #     return MaxiBatch(samples)

    def sample(self, batch_size: int):
        """
        Sample minibatches of exactly `batch_size` transitions
        from a flattened (T * n_envs)-long buffer.
        """
        # self.observations: shape = (T, n_envs, *obs_shape)
        num_steps, n_envs = self.observations.shape[0], self.observations.shape[1]
        total = num_steps * n_envs
        if total < batch_size:
            # Not enough individual transitions to form even one full minibatch
            return MaxiBatch([])

        # ―― 1) Flatten all tensors from shape (T, n_envs, …) → (T*n_envs, …):
        obs_flat = self.observations.reshape(total, *self.observations.shape[2:])
        act_flat = self.actions.reshape(total, *self.actions.shape[2:])
        rew_flat = self.rewards.reshape(total, *self.rewards.shape[2:])
        adv_flat = self.advantages.reshape(total, *self.advantages.shape[2:])
        ret_flat = self.returns.reshape(total, *self.returns.shape[2:])
        epstart_flat = self.episode_starts.reshape(
            total, *self.episode_starts.shape[2:]
        )
        logp_flat = self.log_probs.reshape(total, *self.log_probs.shape[2:])
        val_flat = self.values.reshape(total, *self.values.shape[2:])

        # ―― 2) Randomly permute the “total” indices, keep only full‐batch multiples:
        perm = np.random.permutation(total)
        n_full = total // batch_size
        perm = perm[: n_full * batch_size]
        perm = perm.reshape(n_full, batch_size)

        # ―― 3) Build one RolloutBatch per row of indices:
        samples = []
        for batch_inds in perm:
            obs_batch = obs_flat[batch_inds].numpy()  # shape = (batch_size, *obs_shape)
            act_batch = act_flat[
                batch_inds
            ].numpy()  # shape = (batch_size, *action_shape)
            rew_batch = rew_flat[batch_inds].numpy()  # etc.
            adv_batch = adv_flat[batch_inds].numpy()
            ret_batch = ret_flat[batch_inds].numpy()
            epstart_batch = epstart_flat[batch_inds].numpy()
            logp_batch = logp_flat[batch_inds].numpy()
            val_batch = val_flat[batch_inds].numpy()

            samples.append(
                RolloutBatch(
                    obs_batch,
                    act_batch,
                    rew_batch,
                    adv_batch,
                    ret_batch,
                    epstart_batch,
                    logp_batch,
                    val_batch,
                )
            )

        return MaxiBatch(samples)

    def _get_samples(self, batch_inds: np.ndarray):
        """
        Retrieve a batch of samples given indices.

        :param batch_inds: Indices of transitions to sample.
        :return: RolloutBatch containing the sampled transitions.
        """
        data = (
            self.observations[batch_inds].numpy(),
            self.actions[batch_inds].numpy(),
            self.rewards[batch_inds].numpy(),
            self.advantages[batch_inds].numpy(),
            self.returns[batch_inds].numpy(),
            self.episode_starts[batch_inds].numpy(),
            self.log_probs[batch_inds].numpy(),
            self.values[batch_inds].numpy(),
        )

        return RolloutBatch(*data)

    def __len__(self):
        """
        Return the total number of transitions in the buffer.

        :return: Number of transitions.
        """
        return len(self.observations) * self.n_envs

    def __bool__(self):
        """
        Return whether the buffer contains any transitions.

        :return: True if buffer is not empty, False otherwise.
        """
        return bool(self.observations)

    def save(self, filename="buffer.pkl"):
        """
        Save the buffer to a file.

        :param filename: Path to the file where the buffer will be saved.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)
