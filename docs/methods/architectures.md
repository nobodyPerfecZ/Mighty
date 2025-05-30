# Mighty Architectures
Mighty is made for deep RL, meaning we rely on neural networks for function approximation. You'll find them under the 'mighty_models' keyword in the code. This page should give you a rough overview of their intended use and how to handle them for your experiments.

## Network Structures in Mighty
Mighty networks are based on Torch. 
We implement some basic network architecture building block which can then be combined.
You will usually choose a feature extractor architecture and a head architecture which can be the same or different. 
Furthermore, you can combine two different architectures in the feature extractor. 
You can choose between:
- MLP: standard fully connected networks (flexible structure)
- CNN: 1D or 2D convolutional networks (flexible structure)
- ResNet: a 2D convolutional layer with two residual blocks
- TorchHub model (experimental): loading models from TorchHub

This should cover many standard combinations like a CNN feature extractor with an MLP head.

## Implemented Models
The implemented 'mighty_models' define the prediction patterns for different algorithm classes. 
The DQN model, for example, is initialized to predict Q-values while the PPO model forwards through the policy head when called. 
Both can be based upon the same feature extraction and head structures, of course. 
If we look at the DQN model, we can see it primarily combines different elements to achieve this instead of implementing all of them: 

```python
class DQN(nn.Module):
    """DQN network."""

    def __init__(self, num_actions, obs_size, dueling=False, **kwargs):
        """Initialize the network."""
        super().__init__()
        head_kwargs = {"hidden_sizes": [32, 32]}
        feature_extractor_kwargs = {"obs_shape": obs_size}
        if "head_kwargs" in kwargs:
            head_kwargs.update(kwargs["head_kwargs"])
        if "feature_extractor_kwargs" in kwargs:
            feature_extractor_kwargs.update(kwargs["feature_extractor_kwargs"])

        # Make feature extractor
        self.feature_extractor, self.output_size = make_feature_extractor(
            **feature_extractor_kwargs
        )
        self.dueling = dueling
        self.num_actions = int(num_actions)
        self.obs_size = obs_size
        self.hidden_sizes = head_kwargs["hidden_sizes"]

        # Make policy head
        self.head, self.value, self.advantage = make_q_head(
            self.output_size,
            self.num_actions,
            **head_kwargs,
        )

    def forward(self, x):
        """Forward pass."""
        x = self.feature_extractor(x)
        x = self.head(x)
        advantage = self.advantage(x)
        if self.dueling:
            value = self.value(x)
            x = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            x = advantage
        return x

    def reset_head(self, hidden_sizes=None):
        """Reset the head of the network."""
        if hidden_sizes is None:
            hidden_sizes = self.hidden_sizes
        self.head, self.value, self.advantage = make_q_head(
            self.output_size,
            self.num_actions,
            hidden_sizes,
        )
        self.hidden_sizes = hidden_sizes

    def shrink_weights(self, shrinkage, noise_weight):
        """Shrink weights of the network."""
        params_old = deepcopy(list(self.head.parameters()))
        value_params_old = deepcopy(list(self.value.parameters()))
        adv_params_old = deepcopy(list(self.advantage.parameters()))
        self.reset_head(hidden_sizes=self.hidden_sizes)
        for p_old, p_rand in zip(*[params_old, self.head.parameters()], strict=False):
            p_rand.data = deepcopy(shrinkage * p_old.data + noise_weight * p_rand.data)
        for p_old, p_rand in zip(
            *[adv_params_old, self.advantage.parameters()], strict=False
        ):
            p_rand.data = deepcopy(shrinkage * p_old.data + noise_weight * p_rand.data)
        if self.dueling:
            for p_old, p_rand in zip(
                *[value_params_old, self.value.parameters()], strict=False
            ):
                p_rand.data = deepcopy(
                    shrinkage * p_old.data + noise_weight * p_rand.data
                )

    def __getstate__(self):
        return (
            self.feature_extractor,
            self.head,
            self.advantage,
            self.value,
            self.dueling,
            self.num_actions,
        )

    def __setstate__(self, state):
        self.feature_extractor = state[0]
        self.head = state[1]
        self.advantage = state[2]
        self.value = state[3]
        self.dueling = state[4]
        self.num_actions = state[5]
```
This allows us to have the actual architectures and network structures in central network classes and keeping the model classes quite short. 
As you can see, the DQN class also has additional utility functions like parameter shrinking that can be used in different updates or meta components. 
These are fully optional and can be added as you need them for other components.
Depending on how you structure your model class, you should also revisit the corresponding update to ensure compatibility.

## Changing Network Structure
The MLP and CNN networks have a semi-configurable structure.
Via the algorithm_kwargs, you can specify activations as well as number of layers and units for MLPs and number and kind of convolutions, channels, strides and paddings for CNN.
Hidden sizes, number of channels, stride and padding can be configured per layer for more variation.
Activations, on the other hand, are currently set for the full network.

## When Should I Implement A New Network Class?
Current network classes cover standard cases with some flexibility for MLPs and CNNs. 
The TorchHub option is still being tested and also limited since it's not focused on RL models.
Therefore several relevant options like Transformers still need their own class. 
If you want to use a different architecture than listed here, you should simply make a new class for it and enable its creation via 'make_feature_extractor'.