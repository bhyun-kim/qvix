import torchvision as tv

from qvix.registry import TransformRegistry


@TransformRegistry.register()
def ToTensor() -> tv.transforms.ToTensor:
    """Convert PIL image or numpy array to tensor."""
    return tv.transforms.ToTensor()


@TransformRegistry.register()
def Normalize(mean: list, std: list) -> tv.transforms.Normalize:
    """Normalize a tensor image with mean and standard deviation."""
    return tv.transforms.Normalize(mean, std)


@TransformRegistry.register()
def Resize(size: int) -> tv.transforms.Resize:
    """Resize the input PIL Image to the given size."""
    return tv.transforms.Resize(size)


@TransformRegistry.register()
def RandomCrop(size: int) -> tv.transforms.RandomCrop:
    """Crop the given PIL Image at a random location."""
    return tv.transforms.RandomCrop(size)


@TransformRegistry.register()
def CenterCrop(size: int) -> tv.transforms.CenterCrop:
    """Crop the given PIL Image at the center."""
    return tv.transforms.CenterCrop(size)


@TransformRegistry.register()
def RandomHorizontalFlip(p: float) -> tv.transforms.RandomHorizontalFlip:
    """Horizontally flip the given PIL Image randomly with a given probability."""
    return tv.transforms.RandomHorizontalFlip(p)
