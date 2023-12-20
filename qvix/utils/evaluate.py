from typing import Tuple

import equinox as eqx
import optax
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader

from qvix.utils.step import calculate_step


def evaluate(
    model: eqx.Module,
    testloader: DataLoader,
    key: PRNGKeyArray,
    criterion: optax,
    state: eqx.nn.State,
) -> Tuple[float, float]:
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """

    avg_loss = 0
    avg_acc = 0
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy().astype(int)

        keys = jax.random.split(key, num=x.shape[0])
        keys = jnp.array(keys)

        loss, (acc, _) = calculate_step(model, criterion, keys, x, y, state)

        avg_loss += loss
        avg_acc += acc

    return avg_loss / len(testloader), avg_acc / len(testloader)


@eqx.filter_jit
def compute_accuracy(model: eqx.Module,
                     key: PRNGKeyArray,
                     x: Array,
                     y: Array,
                     inference: bool = True) -> float:
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = model(x, key, inference=inference)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)
