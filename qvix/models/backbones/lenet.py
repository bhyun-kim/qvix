from inspect import signature
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from qvix.registry import BackboneRegistry


@BackboneRegistry.register()
class LeNet5(eqx.Module):
    """LeNet5 <https://ieeexplore.ieee.org/document/726791>.

    
    """

    features: eqx.nn.Sequential
    num_classes: int
    classifier: eqx.nn.Sequential

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):

        keys = jax.random.split(key, 2)

        self.num_classes = num_classes
        self.features = [
            eqx.nn.Conv2d(1, 6, 5, key=keys[0]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2, 2),
            jax.nn.relu,
            eqx.nn.Conv2d(6, 16, 5, key=keys[1]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(2, 2),
        ]

        if num_classes > 0:
            keys = jax.random.split(key, 3)
            self.classifier = [
                eqx.nn.Linear(16 * 5 * 5, 120, key=keys[0]), jax.nn.relu,
                eqx.nn.Linear(120, 84, key=keys[1]), jax.nn.relu,
                eqx.nn.Linear(84, num_classes, key=keys[2])
            ]

    def forward(self,
                x: Array,
                key: Optional[PRNGKeyArray],
                inference: bool = False) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
        """

        kwargs = {"key": key, "inference": inference}

        for layer in self.features:
            x = call_layer(layer, x, **kwargs)

        if self.num_classes > 0:
            x = x.reshape(-1)
            for layer in self.classifier:
                x = call_layer(layer, x, **kwargs)

        return x

    def __call__(self,
                 x: Array,
                 key: Optional[PRNGKeyArray],
                 inference: bool = False) -> Array:
        """Forward with vectorization."""
        return jax.vmap(self.forward, in_axes=(0, 0, None))(x, key, inference)


def call_layer(layer: eqx.Module, x: Array, **kwargs):
    """Call a layer, passing additional keyword arguments if the layer accepts them."""

    layer_params = signature(layer.__call__).parameters
    accepted_kwargs = {k: v for k, v in kwargs.items() if k in layer_params}

    return layer(x, **accepted_kwargs)
