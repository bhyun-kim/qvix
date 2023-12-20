from copy import deepcopy
from typing import Any, Callable, Optional

import equinox as eqx
import torchvision as tv
from jaxtyping import PRNGKeyArray
from torch.utils.data import DataLoader
import optax

from qvix.registry import (BackboneRegistry, DatasetRegistry, 
                           Registry, TransformRegistry)


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


def build_dataloader(dataloader_cfg: dict) -> DataLoader:
    """Build dataloader.
    
    Args:
        dataloader_cfg (dict)
    
    Returns:
        dataloader (torch.utils.data.DataLoader)
    """
    _dataloader_cfg = deepcopy(dataloader_cfg)

    dataset_cfg = _dataloader_cfg.pop("dataset")

    transforms_cfg = dataset_cfg.pop("transforms")
    transforms = build_transforms(transforms_cfg)
    dataset_cfg["transform"] = transforms

    dataset = build_object(dataset_cfg, DatasetRegistry)

    return DataLoader(dataset=dataset, **dataloader_cfg['dataloader'])


def build_transforms(transform_cfgs: list) -> Callable:
    """Build transform.
    
    Args:
        transform_cfg (list)
    
    Returns:
        transform (Callable)
    """
    transforms = []

    for t_cfg in transform_cfgs:
        transform = build_object(t_cfg, TransformRegistry)
        transforms.append(transform)

    return tv.transforms.Compose(transforms)

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

def build_optimizer_chain(optimizer_chain_cfg: dict) -> optax.GradientTransformationExtraArgs:
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
        optimizers.append(build_optax_object(optimizer_cfg))
        
    optimizer = optax.chain(optimizers)
    return optimizer