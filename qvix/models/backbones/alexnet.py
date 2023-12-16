from inspect import signature
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from qvix.models.backbones.base_backbone import call_layer
from qvix.registry import BackboneRegistry


@BackboneRegistry.register()
class AlexNet(eqx.Module):
    """AlexNet backbone.
    
    Reference:
        [1] Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
            Imagenet classification with deep convolutional neural networks. 
            Advances in neural information processing systems, 25.
        
        [2] https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/models/backbones/alexnet.py
    
    """

    features: eqx.nn.Sequential
    num_classes: int
    classifier: eqx.nn.Sequential

    def __init__(self, key: PRNGKeyArray, num_classes: int = -1):

        keys = jax.random.split(key, 5)

        self.num_classes = num_classes
        self.features = [
            eqx.nn.Conv2d(3,
                          64,
                          kernel_size=11,
                          stride=4,
                          padding=2,
                          key=keys[0]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
            eqx.nn.Conv2d(64, 192, kernel_size=5, padding=2, key=keys[1]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
            eqx.nn.Conv2d(192, 384, kernel_size=3, padding=1, key=keys[2]),
            jax.nn.relu,
            eqx.nn.Conv2d(384, 256, kernel_size=3, padding=1, key=keys[3]),
            jax.nn.relu,
            eqx.nn.Conv2d(256, 256, kernel_size=3, padding=1, key=keys[4]),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=3, stride=2),
        ]

        if num_classes > 0:
            key1, key2, key3 = jax.random.split(key, 3)
            self.classifier = [
                eqx.nn.Dropout(0.5),
                eqx.nn.Linear(256 * 6 * 6, 4096, key=key1),
                jax.nn.relu,
                eqx.nn.Dropout(0.5),
                eqx.nn.Linear(4096, 4096, key=key2),
                jax.nn.relu,
                eqx.nn.Linear(4096, num_classes, key=key3),
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
