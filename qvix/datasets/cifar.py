from typing import Any, Callable, Optional

from torchvision.datasets import CIFAR10, CIFAR100

from qvix.registry import DatasetRegistry


@DatasetRegistry.register()
class CIFAR10Dataset(CIFAR10):
    """
    CIFAR10 dataset from torchvision.datasets.CIFAR10
    

    Args (Identical to torchvision.datasets.CIFAR10):
        root (string) : Root directory of dataset where directory cifar-10-batches-py exists 
            or will be saved to if download is set to True.
        train (bool, optional) : If True, creates dataset from training set, otherwise creates
            from test set.
        transform (callable, optional) : A function/transform that takes in an PIL image and 
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional) : A function/transform that takes in the target
            and transforms it.
        download (bool, optional) : If true, downloads the dataset from the internet and puts
            it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Get item from dataset.
        
        Args:
            index (int) : Index
        """
        img, target = super().__getitem__(index)
        return img, target


@DatasetRegistry.register()
class CIFAR100Dataset(CIFAR100):
    """
    CIFAR100 dataset from torchvision.datasets.CIFAR100

    Args (Identical to torchvision.datasets.CIFAR100):
        root (string) : Root directory of dataset where directory cifar-100-batches-py exists 
            or will be saved to if download is set to True.
        train (bool, optional) : If True, creates dataset from training set, otherwise creates 
            from test set.
        transform (callable, optional) : A function/transform that takes in an PIL image and 
            returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional) : A function/transform that takes in the target 
            and transforms it.
        download (bool, optional) : If true, downloads the dataset from the internet and puts 
            it in root directory. If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False) -> None:
        super().__init__(root,
                         train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Get item from dataset.

        Args:
            index (int) : Index
        """
        img, target = super().__getitem__(index)
        return img, target
