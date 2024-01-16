from typing import Optional

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
from jaxtyping import Array, PRNGKeyArray

from qvix.registry import BackboneRegistry


@BackboneRegistry.register()
class LeNet5(eqx.Module):
    """LeNet5 <https://ieeexplore.ieee.org/document/726791>.    
    """
    features: nn.Sequential
    num_classes: int
    classifier: nn.Sequential

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):

        keys = jax.random.split(key, 2)

        self.num_classes = num_classes
        self.features = nn.Sequential([
            nn.Conv2d(1, 6, 5, key=keys[0]),
            nn.Lambda(jnn.relu),
            nn.MaxPool2d(2, 2),
            nn.Lambda(jnn.relu),
            nn.Conv2d(6, 16, 5, key=keys[1]),
            nn.Lambda(jnn.relu),
            nn.MaxPool2d(2, 2),
        ])

        if num_classes > 0:
            keys = jax.random.split(key, 3)
            self.classifier = nn.Sequential([
                nn.Linear(16 * 5 * 5, 120, key=keys[0]),
                nn.Lambda(jnn.relu),
                nn.Linear(120, 84, key=keys[1]),
                nn.Lambda(jnn.relu),
                nn.Linear(84, num_classes, key=keys[2])
            ])

    def __call__(self,
                 x: Array,
                 state: nn.State = None,
                 key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
            key (PRNGKeyArray): ignored; only for compatibility with other backbones.
            state (nn.State): ignored; only for compatibility with other backbones.
        """

        out = self.features(x)
        out = out.reshape(-1)
        out = self.classifier(out)
        return out, state
