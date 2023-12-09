from copy import deepcopy
from typing import Any, Callable, Optional

import equinox as eqx
import torchvision as tv
from jaxtyping import Array, PRNGKeyArray
from torch.utils.data import DataLoader

import jax.numpy as jnp

from qvix.registry import (BackboneRegistry, DatasetRegistry,
                                LossRegistry, Registry, TransformRegistry)


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
    name = cfg.pop("name")

    if key is not None:
        return registry(name)(key, **cfg)
    else:
        return registry(name)(**cfg)


def build_dataloader(dataloader_cfg: dict) -> DataLoader:
    """Build dataloader.
    
    Args:
        dataloader_cfg (dict)
    
    Returns:
        dataloader (torch.utils.data.DataLoader)
    """
    dataset_cfg = dataloader_cfg.pop("dataset")
    # build transforms
    transforms_cfg = dataset_cfg.pop("transforms")
    transforms = []
    for transform in transforms_cfg:
        name = transform.pop("name")
        transforms.append(TransformRegistry(name)(**transform))

    transforms = tv.transforms.Compose(transforms)

    dataset_name = dataset_cfg.pop("name")
    dataset = DatasetRegistry(dataset_name)(**dataset_cfg,
                                            transform=transforms)

    return DataLoader(dataset=dataset, **dataloader_cfg['dataloader'])


def build_backbone(backbone_cfg: dict, key: PRNGKeyArray) -> eqx.Module:
    """Build backbone.
    
    Args:
        key (jax.random.PRNGKey)
        backbone_cfg (dict)
    
    Returns:
        backbone (eqx.Module)
    """
    return build_object(backbone_cfg, BackboneRegistry, key)


def build_transforms(transform_cfgs: list) -> Callable:
    """Build transform.
    
    Args:
        transform_cfg (list)
    
    Returns:
        transform (Callable)
    """
    transforms = []

    for transform in transform_cfgs:
        name = transform.pop("name")
        transforms.append(TransformRegistry(name)(**transform))

    return tv.transforms.Compose(transforms)


def build_dataset(dataset_cfg: dict) -> Any:
    """Build dataset.
    
    Args:
        dataset_cfg (dict)
    
    Returns:
        dataset (Any)
    """
    dataset_cfg["transforms"] = build_transforms(dataset_cfg["transforms"])
    return build_object(dataset_cfg, DatasetRegistry)


def calculate_loss(model: eqx.Module,
                   loss_cfg: dict,
                   key: PRNGKeyArray,
                   x: Array,
                   labels: Array,
                   inference: bool = False) -> Array:
    """Forward with optax.softmax_cross_entropy."""

    logits = model(x, key=key, inference=inference)

    _loss_cfg = deepcopy(loss_cfg)
    loss_name = _loss_cfg.pop('name')
    return LossRegistry(loss_name)(logits, labels, **_loss_cfg)

def calculate_step(model: eqx.Module,
                   loss_cfg: dict,
                   key: PRNGKeyArray,
                   x: Array,
                   y: Array,
                   inference: bool = False) -> Array:
    """Forward with optax.softmax_cross_entropy."""

    logits = model(x, key=key, inference=inference)

    pred_y = jnp.argmax(logits, axis=1)
    acc = jnp.mean(y == pred_y)

    _loss_cfg = deepcopy(loss_cfg)
    loss_name = _loss_cfg.pop('name')
    loss = LossRegistry(loss_name)(logits, y, **_loss_cfg)
    return loss, acc
