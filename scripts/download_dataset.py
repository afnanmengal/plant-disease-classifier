"""Download the public PlantVillage dataset using KaggleHub."""

from pathlib import Path

import kagglehub

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def find_class_directory(dataset_path: str | Path) -> Path:
    """Find the directory whose immediate subdirectories contain class images."""
    dataset_path = Path(dataset_path)
    candidates: list[tuple[int, int, Path]] = []
    for candidate in (dataset_path, *dataset_path.rglob("*")):
        if not candidate.is_dir():
            continue
        class_directories = [path for path in candidate.iterdir() if path.is_dir()]
        image_classes = sum(
            any(image.suffix.lower() in IMAGE_SUFFIXES for image in class_dir.iterdir())
            for class_dir in class_directories
        )
        if image_classes >= 2:
            candidates.append((image_classes, -len(class_directories), candidate))
    if not candidates:
        raise FileNotFoundError(f"No class-folder image dataset found under {dataset_path}")
    return max(candidates, key=lambda result: result[:2])[2]


if __name__ == "__main__":
    dataset_path = kagglehub.dataset_download("emmarex/plantdisease")
    class_directory = find_class_directory(dataset_path)
    print(f"PlantVillage class folders: {class_directory}")