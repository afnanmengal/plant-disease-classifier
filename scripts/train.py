"""Command-line entry point for classifier training."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from plant_disease.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PlantVillage disease classifier")
    parser.add_argument(
        "--data-dir", required=True, help="Folder containing one subfolder per class"
    )
    parser.add_argument("--output", default="models/plant_disease_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    train_model(
        args.data_dir,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        device=args.device,
        pretrained=not args.no_pretrained,
        num_workers=args.num_workers,
    )
    print(f"Best checkpoint saved to {args.output}")


if __name__ == "__main__":
    main()