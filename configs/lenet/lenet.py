iterations = 10000
batch_size = 64
work_dir = "work_dir/lenet"
seed = 0

checkpoint_interval = 1000
log_interval = 100
validate_interval = 1000

model = dict(
    name="LeNet5",
    num_classes=10,
)

load_from = None
resume_from = None
initialization = None

loss = dict(name="softmax_cross_entropy_with_integer_labels")

train_transforms = [
    dict(name="Resize", size=32),
    dict(name="ToTensor"),
    dict(name="Normalize", mean=[
        0.1307,
    ], std=[
        0.3081,
    ])
]

test_transforms = [
    dict(name="Resize", size=32),
    dict(name="ToTensor"),
    dict(name="Normalize", mean=[
        0.1307,
    ], std=[
        0.3081,
    ])
]

train_loader = dict(
    dataset=dict(
        name="MNIST",
        root="data\MNIST",
        train=True,
        transforms=train_transforms,
        download=True,
    ),
    dataloader=dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    ),
)

test_loader = dict(
    dataset=dict(
        name="MNIST",
        root="data\MNIST",
        train=False,
        transforms=test_transforms,
        download=True,
    ),
    dataloader=dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    ),
)

optimizer = dict(name="adam", learning_rate=3e-4)
