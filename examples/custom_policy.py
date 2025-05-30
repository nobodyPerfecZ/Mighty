"""UCB exploration for DQN."""

from __future__ import annotations

import numpy as np

from mighty.mighty_exploration.mighty_exploration_policy import MightyExplorationPolicy


class QValueUCB(MightyExplorationPolicy):
    """Exploration via UCB for DQN."""

    def __init__(
        self,
        algo,
        model,
        constant=2,
    ):
        """Initialize UCB.

        :param algo: algorithm name
        :param func: policy function
        :param constant: c constant for UCB
        :return:
        """
        super().__init__(algo, model)
        self.c = constant
        self.action_selected_count = np.zeros(model.num_actions)

    def explore(self, s, return_logp, metrics):
        """Explore.

        :param s: state
        :param return_logp: return logprobs
        :param metrics: metrics dictionary
        :return: action or (action, logprobs)
        """
        # Get Q-values
        _, qvals = self.sample_action(s)
        # Calculate UCB bonus
        ucb_bonus = self.c*np.sqrt(np.log(metrics["step"] + 1e-4)/(self.action_selected_count + 1e-4))
        # Add bonus and selection actions
        ucb_actions = np.argmax(qvals + ucb_bonus)
        # Update action counter
        for action in ucb_actions:
            self.action_selected_count[action] += 1
        return (ucb_actions, qvals) if return_logp else ucb_actions