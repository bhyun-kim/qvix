from inspect import signature
from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray


def call_layer(layer: eqx.Module, x: Array, **kwargs):
    """Call a layer, passing additional keyword arguments if the layer accepts them."""

    layer_params = signature(layer.__call__).parameters
    accepted_kwargs = {k: v for k, v in kwargs.items() if k in layer_params}

    return layer(x, **accepted_kwargs)


class BaseBackbone(eqx.Module):
    """Base class for backbones."""

    def __init__(self):
        super().__init__()

    def forward(self,
                x: Array,
                key: Optional[PRNGKeyArray],
                inference: bool = False) -> Array:
        """Forward pass.

        Args:
            x (Array): Input tensor.
        """
        raise NotImplementedError

    def __call__(self,
                 x: Array,
                 key: Optional[PRNGKeyArray],
                 inference: bool = False) -> Array:
        """Forward with vectorization."""
        return jax.vmap(self.forward, in_axes=(0, 0, None))(x, key, inference)