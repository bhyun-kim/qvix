from typing import Any

from torchvision.datasets import ImageNet

from qvix.registry import DatasetRegistry


@DatasetRegistry.register()
class ImageNetDataset(ImageNet):
    """
    ImageNet dataset from torchvision.datasets.ImageNet
    
    Args (Identical to torchvision.datasets.ImageNet):
        root (string) : Root directory of the ImageNet Dataset.
        split (string, optional) : The dataset split, supports train, or val.
        transform (callable, optional) : A function/transform that takes in an PIL image and 
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional) : A function/transform that takes in the target 
            and transforms it.
        loader : A function to load an image given its path.
    """

    def __init__(self, root: str, split: str = "train", **kwargs: Any):
        super().__init__(root, split=split, **kwargs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Get item from dataset.
        
        Args:
            index (int) : Index
        """
        img, target = super().__getitem__(index)
        return img, target
