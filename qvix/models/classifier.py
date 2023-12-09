from typing import Optional

import equinox as eqx
import jax
from jaxtyping import Array, PRNGKeyArray

from qvix.registry import ModelRegistry


@ModelRegistry.register()
class ImageClassifier(eqx.Module):
    """ImageClassifier.
    
    Args:
        num_classes (int)
    """
    backbone: eqx.Module

    def __init__(self, backbone: eqx.Module, loss: eqx.Module) -> None:

        self.backbone = backbone

    def __call__(self,
                 x: Array,
                 key: Optional[PRNGKeyArray],
                 inference: bool = False) -> Array:
        """Forward."""
        return jax.vmap(self.backbone, in_axes=(0, 0, None))(x, key, inference)
