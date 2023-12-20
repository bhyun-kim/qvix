from copy import deepcopy
from typing import Callable

import torchvision as tv
from torch.utils.data import DataLoader

from qvix.registry import (DatasetRegistry, TransformRegistry)
from qvix.builder.builder import build_object


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

    if 'backend' in dataset_cfg and dataset_cfg['backend'] is 'qvix':
        dataset = build_object(dataset_cfg, DatasetRegistry)
    else:
        dataset = getattr(tv.datasets, dataset_cfg['name'])(**dataset_cfg)

    return DataLoader(dataset=dataset, **dataloader_cfg['dataloader'])


def build_transforms(transform_cfgs: list) -> Callable:
    """Build transform.
    
    Args:
        transform_cfg (list)
    
    Returns:
        transform (Callable)
    """
    _transform_cfg = deepcopy(transform_cfgs)

    transforms = []
    for t_cfg in _transform_cfg:
        name = t_cfg.pop('name')

        if 'backend' in t_cfg and t_cfg['backend'] is 'qvix':
            transforms.append(build_object(t_cfg, TransformRegistry))
        else: 
            transforms.append(getattr(tv.transforms, name)(**t_cfg))

    return tv.transforms.Compose(transforms)