from abc import ABC, abstractmethod

import numpy as np


class MightyBuffer(ABC):
    @abstractmethod
    def add(self, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __bool__(self):
        pass

    def seed(self, seed: int):
        """Set random seed."""
        self.rng = np.random.default_rng(seed)
