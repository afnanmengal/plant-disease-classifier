import pytest
import torch

from plant_disease.training import train_model


def test_cuda_training_rejects_unsupported_gpu_architecture(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])

    with pytest.raises(RuntimeError, match="does not support sm_120"):
        train_model(tmp_path / "missing-data", tmp_path / "model.pt", pretrained=False)