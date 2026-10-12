"""ResNet18 model construction, training, and single-image inference."""

from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from plant_disease.data import IMAGENET_MEAN, IMAGENET_STD


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build ResNet18 with a classifier head matching the dataset classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return model


def load_checkpoint(
    checkpoint_path: str | Path, device: torch.device | str = "cpu"
) -> tuple[nn.Module, list[str], int]:
    """Restore a saved classifier and its label metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = create_model(len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, int(checkpoint["image_size"])


def predict_image(
    model: nn.Module,
    image_path: str | Path | BinaryIO,
    class_names: list[str],
    image_size: int = 224,
    device: torch.device | str = "cpu",
) -> tuple[str, float]:
    """Return the top predicted class and its softmax confidence."""
    return predict_image_top_k(model, image_path, class_names, image_size, device, top_k=1)[0]


def predict_image_top_k(
    model: nn.Module,
    image_path: str | Path | BinaryIO,
    class_names: list[str],
    image_size: int = 224,
    device: torch.device | str = "cpu",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Return the highest-scoring classes and their softmax probabilities."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    image_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    with Image.open(image_path) as image:
        tensor = image_transform(image.convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores, indices = probabilities.topk(min(top_k, len(class_names)))
    return [
        (class_names[int(index)], float(score)) for index, score in zip(indices, scores)
    ]