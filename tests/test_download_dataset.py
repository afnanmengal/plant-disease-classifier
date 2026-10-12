from pathlib import Path

from scripts.download_dataset import find_class_directory


def test_finds_inner_class_root_when_dataset_has_duplicate_wrapper(tmp_path: Path):
    outer = tmp_path / "PlantVillage"
    nested = outer / "PlantVillage"
    for class_name in ("Class_A", "Class_B"):
        (outer / class_name).mkdir(parents=True)
        (outer / class_name / "leaf.jpg").touch()
        (nested / class_name).mkdir(parents=True)
        (nested / class_name / "leaf.jpg").touch()

    assert find_class_directory(outer) == nested