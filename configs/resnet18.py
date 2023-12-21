iterations = 78200
batch_size = 128
work_dir = "work_dir/resnet/cifar10"
seed = 0

checkpoint_interval = 3910
log_interval = 100
validate_interval = 3910

model = dict(
    name="ResNet18",
    num_classes=10,
)

load_from = None
resume_from = None
initialization = None

loss = dict(name="softmax_cross_entropy_with_integer_labels")

train_transforms = [
    # dict(name='ColorJitter',
    #      brightness=0.2,
    #      contrast=0.2,
    #      saturation=0.2,
    #      hue=0.2),
    dict(name="RandomCrop", size=32, padding=4),
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
    dataset=dict(name="CIFAR10",
                 root="data/cifar10",
                 train=True,
                 transforms=train_transforms,
                 download=True),
    dataloader=dict(batch_size=batch_size, shuffle=True, num_workers=4),
)

test_loader = dict(
    dataset=dict(name="CIFAR10",
                 root="data/cifar10",
                 train=False,
                 transforms=test_transforms,
                 download=True),
    dataloader=dict(batch_size=128, shuffle=False, num_workers=4),
)

optimizer_chain = [
    # dict(name="add_decayed_weights", weight_decay=5e-4),
    dict(name="adamw",
         momentum=0.9,
         scheduler=dict(name="cosine_decay_schedule",
                        init_value=0.001,
                        decay_steps=iterations))
]
