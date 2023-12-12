model = dict(
    name="AlexNet",
    num_classes=1000,
)

loss = dict(name="SoftmaxCrossEntropyLoss"),

train_transforms = [
    dict(name="Resize", size=256),
    dict(name="RandomCrop", size=224),
    dict(name="RandomHorizontalFlip", p=0.5),
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
]

test_transforms = [
    dict(name="Resize", size=256),
    dict(name="CenterCrop", size=224),
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
]

train_loader = dict(
    dataset=dict(
        name="ImageNetDataset",
        root="data\ImageNet",
        train=True,
        transforms=train_transforms,
        download=True,
    ),
    dataloader=dict(
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ),
)

test_loader = dict(
    dataset=dict(
        name="ImageNetDataset",
        root="data\ImageNet",
        train=False,
        transforms=test_transforms,
        download=True,
    ),
    dataloader=dict(
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ),
)

optimizer = dict(
    name="Adam",
    learning_rate=0.01,
)

epochs = 10
batch_size = 128