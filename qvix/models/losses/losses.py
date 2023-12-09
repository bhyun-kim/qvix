import chex
import equinox as eqx
import optax

from qvix.registry import LossRegistry


@LossRegistry.register()
def SoftmaxCrossEntropyLoss(logits: chex.Array,
                            labels: chex.Array,
                            reduce: str = 'mean'):
    """Softmax cross entropy loss."""
    """Forward with optax.softmax_cross_entropy."""

    losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    if reduce == 'mean':
        return losses.mean()

    elif reduce == 'sum':
        return losses.sum()

    elif reduce == 'none':
        return losses
