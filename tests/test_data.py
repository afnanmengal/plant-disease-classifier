from pathlib import Path

import pytest

from plant_disease.data import build_dataloaders


def test_missing_dataset_path_has_actionable_error(tmp_path: Path):
    with pytest.raises(ValueError, match="Image directory does not exist"):
        build_dataloaders(tmp_path / "missing")


def test_invalid_validation_fraction_is_rejected(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="validation_fraction"):
        build_dataloaders(tmp_path, validation_fraction=1.0)