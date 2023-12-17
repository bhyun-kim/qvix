iterations = 78200
batch_size = 128
work_dir = "work_dir/resnet/cifar10"
seed = 0

checkpoint_interval = 3910
log_interval = 1
validate_interval = 3910

load_from = None
resume_from = None

model = dict(
    name="ResNet18",
    num_classes=10,
)

loss = dict(name="SoftmaxCrossEntropyLoss")

train_transforms = [
    dict(name="Resize", size=40),
    dict(name="RandomCrop", size=32),
    dict(name="RandomHorizontalFlip", p=0.5),
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.4914, 0.4822, 0.4465],
         std=[0.2023, 0.1994, 0.2010]),
]

test_transforms = [
    dict(name="ToTensor"),
    dict(name="Normalize",
         mean=[0.4914, 0.4822, 0.4465],
         std=[0.2023, 0.1994, 0.2010]),
]

train_loader = dict(
    dataset=dict(name="CIFAR10Dataset",
                 root="data/cifar10",
                 train=True,
                 transforms=train_transforms,
                 download=True),
    dataloader=dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
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
        num_workers=4
    ),
)

# optimizer = dict(
#     name="SGD",
#     momentum=0.9,
#     scheduler=dict(
#         init_value=0.1,
#         peak_value=0.1,
#         name="WarmupCosineDecay",
#         warmup_steps=1,
#         decay_steps=iterations
#     ),
# )

optimizer_chain = [
    dict(
        name='Add_decayed_weights',
        weight_decay=0.0, 
    ),
    dict(
    name="SGD",
    momentum=0.9,
    scheduler=dict(
        init_value=0.1,
        peak_value=0.1,
        name="WarmupCosineDecay",
        warmup_steps=1,
        decay_steps=iterations
    ),
)
]

