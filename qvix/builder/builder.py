from copy import deepcopy
from typing import Any, Optional

import equinox as eqx
from jaxtyping import PRNGKeyArray
import optax

import jax.numpy as jnp

from qvix.registry import (BackboneRegistry, Registry)


def build_object(cfg: dict,
                 registry: Registry,
                 key: Optional[PRNGKeyArray] = None) -> Any:
    """Basic builder.
    
    Args:
        key (jax.random.PRNGKey)
        cfg (dict)
    
    Returns:
        Any
    """
    _cfg = deepcopy(cfg)
    name = _cfg.pop("name")

    if key is not None:
        return registry(name)(key, **_cfg)
    else:
        return registry(name)(**_cfg)


def build_backbone(model_cfg: dict, key: PRNGKeyArray) -> eqx.Module:
    """Build model.
    
    Args:
        model_cfg (dict)
        key (jax.random.PRNGKey)
    
    Returns:
        model (eqx.Module)
    """
    _model_cfg = deepcopy(model_cfg)

    model_name = _model_cfg.pop("name")
    model = BackboneRegistry(model_name)
    model, model_state = eqx.nn.make_with_state(model)(key, **_model_cfg)
    return model, model_state


def build_optax_object(cfg: dict) -> Any:
    """Build optax object.
    
    Args:
        cfg (dict)
    
    Returns:
        Any
    """
    _cfg = deepcopy(cfg)
    name = _cfg.pop("name")

    return getattr(optax, name)(**_cfg)


def build_optimizer(
        optimizer_cfg: dict) -> optax.GradientTransformationExtraArgs:
    """Build optimizer.
    
    Args:
        optimizer_cfg (dict)
        model (eqx.Module)
    
    Returns:
        optimizer (optax.GradientTransformationExtraArgs)
    """
    if 'scheduler' in optimizer_cfg:
        scheduler_cfg = optimizer_cfg.pop('scheduler')
        scheduler = build_optax_object(scheduler_cfg)
        optimizer_cfg['learning_rate'] = scheduler
    optimizer = build_optax_object(optimizer_cfg)
    return optimizer


def build_optimizer_chain(
        optimizer_chain_cfg: dict) -> optax.GradientTransformationExtraArgs:
    """Build optimizer chain.
    
    Args:
        optimizer_chain_cfg (dict)
        model (eqx.Module)
    
    Returns:
        optimizer (optax.GradientTransformationExtraArgs)
    """
    _optimizer_chain_cfg = deepcopy(optimizer_chain_cfg)
    optimizers = []

    for optimizer_cfg in _optimizer_chain_cfg:
        if 'scheduler' in optimizer_cfg or 'learning_rate' in optimizer_cfg:
            optimizers.append(build_optimizer(optimizer_cfg))
        else:
            optimizers.append(build_optax_object(optimizer_cfg))

    optimizer = optax.chain(optimizers)
    return optimizer


def build_loss_function(loss_cfg: dict) -> Any:
    """Build loss function.
    
    Args:
        loss_cfg (dict)
    
    Returns:
        Any
    """
    return OptaxLossFunction(loss_cfg)


class OptaxLossFunction(object):
    """Optax loss function.
    
    Args:
        loss_cfg (dict)
    """

    def __init__(self, loss_cfg: dict, reduce: str = "mean") -> None:
        self._loss_cfg = deepcopy(loss_cfg)
        self._reduce = reduce
        self._loss_name = self._loss_cfg.pop('name')
        self._loss = getattr(optax, self._loss_name)

    def __call__(self, *args) -> jnp.ndarray:
        losses = self._loss(*args, **self._loss_cfg)
        if self._reduce == "mean":
            return jnp.mean(losses)
        elif self._reduce == "sum":
            return jnp.sum(losses)
        elif self._reduce == "none":
            return losses

    def __repr__(self) -> str:
        return f"OptaxLossFunction(loss_name={self._loss_name})"
