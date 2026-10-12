"""ImageFolder data loading and reproducible train/validation splitting."""

from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training augmentation and deterministic validation transforms."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def build_dataloaders(
    image_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    validation_fraction: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load class-folder images and return stratified train/validation loaders."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    train_transform, validation_transform = build_transforms(image_size)
    train_dataset = datasets.ImageFolder(image_dir, transform=train_transform)
    validation_dataset = datasets.ImageFolder(image_dir, transform=validation_transform)
    if len(train_dataset.classes) < 2:
        raise ValueError("The dataset must contain at least two class subdirectories")

    indices = list(range(len(train_dataset)))
    labels = train_dataset.targets
    try:
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=validation_fraction,
            random_state=seed,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Each class needs enough images for a stratified train/validation split"
        ) from error

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, validation_loader, train_dataset.classes