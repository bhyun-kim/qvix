from copy import deepcopy
from typing import Callable

import torchvision as tv
from torch.utils.data import DataLoader


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

    name = dataset_cfg.pop('name')
    dataset = getattr(tv.datasets, name)(**dataset_cfg)

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
        transforms.append(getattr(tv.transforms, name)(**t_cfg))

    return tv.transforms.Compose(transforms)