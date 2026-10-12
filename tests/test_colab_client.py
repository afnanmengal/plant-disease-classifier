import requests

from plant_disease.colab_client import predict_with_colab


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "device": "Tesla T4",
            "predictions": [
                {"class_name": "Tomato_healthy", "confidence": 0.8},
                {"class_name": "Tomato_Late_blight", "confidence": 0.15},
            ],
        }


def test_colab_client_sends_image_and_authenticates(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    predictions, device = predict_with_colab(
        "https://leaf-tunnel.example/", "test-token", b"image", "leaf.png", "image/png"
    )

    assert captured["url"] == "https://leaf-tunnel.example/predict"
    assert captured["headers"] == {"X-API-Key": "test-token"}
    assert captured["files"]["file"] == ("leaf.png", b"image", "image/png")
    assert predictions[0] == ("Tomato_healthy", 0.8)
    assert device == "Tesla T4"


def test_colab_client_requires_api_credentials():
    try:
        predict_with_colab("", "", b"image", "leaf.png", "image/png")
    except ValueError as error:
        assert "API URL and token" in str(error)
    else:
        raise AssertionError("Missing credentials should be rejected")