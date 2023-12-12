iterations = int((1300 * 1000 // 128) * 200)
batch_size = 128
work_dir = "work_dir/lenet"
seed = 0

checkpoint_interval = 1000
log_interval = 100
validate_interval = 1000

load_from = None
resume_from = None

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
        split="train",
        transforms=train_transforms,
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
        split="val",
        transforms=test_transforms,
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
    learning_rate=3e-4,
)
