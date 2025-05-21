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
    ):
        """
        Initialize a batch of rollout transitions.

        :param observations: Numpy array of observations.
        :param actions: Numpy array of actions.
        :param rewards: Numpy array of rewards.
        :param advantages: Numpy array of advantages.
        :param returns: Numpy array of returns.
        :param episode_starts: Numpy array indicating episode starts.
        :param log_probs: Numpy array of log probabilities.
        :param values: Numpy array of value estimates.
        """
        self.observations = torch.from_numpy(observations.astype(np.float32)).unsqueeze(
            0
        )
        self.actions = torch.from_numpy(actions.astype(np.float32)).unsqueeze(0)
        self.rewards = torch.from_numpy(rewards.astype(np.float32)).unsqueeze(0)
        self.advantages = torch.from_numpy(advantages.astype(np.float32)).unsqueeze(0)
        self.returns = torch.from_numpy(returns.astype(np.float32)).unsqueeze(0)
        self.episode_starts = torch.from_numpy(
            episode_starts.astype(np.float32)
        ).unsqueeze(0)
        self.log_probs = torch.from_numpy(log_probs.astype(np.float32)).unsqueeze(0)
        self.values = torch.from_numpy(values.astype(np.float32)).unsqueeze(0)

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
        device: str = "auto",
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

    # FIXME: loads of code duplication here, just call super().reset() first
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

        if len(self.observations) == 0:
            self.observations = rollout_batch.observations
            self.actions = rollout_batch.actions
            self.rewards = rollout_batch.rewards
            self.advantages = rollout_batch.advantages
            self.returns = rollout_batch.returns
            self.episode_starts = rollout_batch.episode_starts
            self.log_probs = rollout_batch.log_probs
            self.values = rollout_batch.values
        else:
            self.observations = torch.cat(
                (self.observations, rollout_batch.observations)
            )
            self.actions = torch.cat((self.actions, rollout_batch.actions))
            self.rewards = torch.cat((self.rewards, rollout_batch.rewards))
            self.advantages = torch.cat((self.advantages, rollout_batch.advantages))
            self.returns = torch.cat((self.returns, rollout_batch.returns))
            self.episode_starts = torch.cat(
                (self.episode_starts, rollout_batch.episode_starts)
            )
            self.log_probs = torch.cat((self.log_probs, rollout_batch.log_probs))
            self.values = torch.cat((self.values, rollout_batch.values))

    def sample(self, batch_size: int):
        """
        Sample mini-batches of transitions from the buffer.

        :param batch_size: Number of transitions per mini-batch.
        :return: List of RolloutBatch samples.
        """
        # FIXME: maybe truncate batch size instead?
        hangover = len(self.observations) % batch_size
        indices = np.random.permutation(len(self.observations))
        indices = (
            indices[:-hangover].reshape(-1, batch_size).tolist()
            + indices[-hangover:].tolist()
        )
        samples = []
        for ind in indices:
            samples.append(self._get_samples(ind))
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
