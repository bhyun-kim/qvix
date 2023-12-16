from copy import deepcopy
from typing import Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, PRNGKeyArray

from qvix.registry import LossRegistry


def calculate_step(model: eqx.Module, loss_cfg: dict, key: PRNGKeyArray,
                   x: Array, y: Array, state: nn.State) -> Array:
    """Forward with optax.softmax_cross_entropy."""

    model = jax.vmap(model, axis_name="batch", in_axes=(0, None, 0), out_axes=(0, None))

    logits, state = model(x, state, key)

    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)

    _loss_cfg = deepcopy(loss_cfg)
    loss_name = _loss_cfg.pop('name')
    loss = LossRegistry(loss_name)(logits, y, **_loss_cfg)
    return loss, (acc, state)


@eqx.filter_jit
def make_step(model: eqx.Module, model_state: nn.State,
              opt_state: optax.OptState,
              optimizer: optax.GradientTransformation, key: PRNGKeyArray,
              loss_cfg: dict, x: Array,
              y: Array) -> Tuple[eqx.Module, nn.State, optax.OptState, float]:
    keys = jax.random.split(key, num=x.shape[0])
    keys = jax.numpy.array(keys)

    (loss, (acc, model_state)), grads = eqx.filter_value_and_grad(
        calculate_step, has_aux=True)(model, loss_cfg, keys, x, y, model_state)
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, model_state, opt_state, loss, acc
