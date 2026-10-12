"""Training loop for the PlantVillage image classifier."""

from pathlib import Path

import torch
from torch import nn

from plant_disease.data import build_dataloaders
from plant_disease.model import create_model


def train_model(
    image_dir: str | Path,
    output_path: str | Path,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    image_size: int = 224,
    pretrained: bool = True,
    num_workers: int = 0,
    device: str = "cuda",
) -> dict[str, float]:
    """Train the classifier and save the checkpoint with best validation accuracy."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-capable PyTorch GPU is available")
        capability = torch.cuda.get_device_capability(device)
        required_architecture = f"sm_{capability[0]}{capability[1]}"
        if required_architecture not in torch.cuda.get_arch_list():
            raise RuntimeError(
                f"The installed PyTorch build does not support {required_architecture}. "
                "Install the CUDA 12.8 build from requirements.txt."
            )
    train_loader, validation_loader, class_names = build_dataloaders(
        image_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    model = create_model(len(class_names), pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    loss_function = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    history: dict[str, float] = {}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * labels.size(0)

        model.eval()
        correct = 0
        sample_count = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == labels).sum())
                sample_count += labels.size(0)

        validation_accuracy = correct / sample_count
        history = {
            "train_loss": train_loss_total / len(train_loader.dataset),
            "validation_accuracy": validation_accuracy,
        }
        print(
            f"Epoch {epoch + 1}/{epochs} | loss {history['train_loss']:.4f} "
            f"| validation accuracy {validation_accuracy:.4f}"
        )
        if validation_accuracy >= best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "image_size": image_size,
                    "validation_accuracy": validation_accuracy,
                },
                output_path,
            )

    return history