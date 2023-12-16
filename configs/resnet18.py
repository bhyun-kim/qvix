iterations = 10000
batch_size = 128
work_dir = "work_dir/resnet/cifar10"
seed = 0

checkpoint_interval = 1000
log_interval = 1
validate_interval = 1000

load_from = None
resume_from = None

model = dict(
    name="ResNet18",
    num_classes=10,
)

loss = dict(name="SoftmaxCrossEntropyLoss")

train_transforms = [
    dict(name="Resize", size=36),
    dict(name="RandomCrop", size=32),
    dict(name="RandomHorizontalFlip", p=0.5),
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
]

test_transforms = [
    dict(name="Resize", size=36),
    dict(name="CenterCrop", size=32),
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]),
]

train_loader = dict(
    dataset=dict(name="CIFAR10Dataset",
                 root="data/cifar10",
                 train=True,
                 transforms=train_transforms,
                 download=True),
    dataloader=dict(
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ),
)

test_loader = dict(
    dataset=dict(name="CIFAR10Dataset",
                 root="data/cifar10",
                 train=False,
                 transforms=test_transforms,
                 download=True),
    dataloader=dict(
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ),
)

optimizer = dict(
    name="Adam",
    learning_rate=3e-4,
)
