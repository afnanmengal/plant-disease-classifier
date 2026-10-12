import torch
from PIL import Image
from torch import nn

from plant_disease.model import create_model, predict_image_top_k


def test_model_outputs_one_score_per_class():
    model = create_model(num_classes=4, pretrained=False)
    output = model(torch.zeros(2, 3, 224, 224))

    assert output.shape == (2, 4)


def test_only_classifier_head_is_trainable():
    model = create_model(num_classes=3, pretrained=False)
    frozen_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("fc.")
    ]

    assert all(not parameter.requires_grad for parameter in frozen_parameters)
    assert all(parameter.requires_grad for parameter in model.fc.parameters())


def test_prediction_returns_classes_ranked_by_probability(tmp_path):
    image_path = tmp_path / "leaf.png"
    Image.new("RGB", (32, 32), color="green").save(image_path)

    class FixedModel(nn.Module):
        def forward(self, images):
            return torch.tensor([[0.0, 3.0, 1.0]], device=images.device)

    predictions = predict_image_top_k(
        FixedModel(), image_path, ["healthy", "rust", "blight"], top_k=2
    )

    assert [name for name, _ in predictions] == ["rust", "blight"]
    assert predictions[0][1] > predictions[1][1]