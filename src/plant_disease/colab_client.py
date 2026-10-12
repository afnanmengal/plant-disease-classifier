"""HTTP client for the optional Google Colab GPU inference service."""

from typing import Any

import requests


def predict_with_colab(
    api_url: str,
    api_token: str,
    image_bytes: bytes,
    filename: str,
    content_type: str,
    top_k: int = 3,
    timeout: float = 60,
) -> tuple[list[tuple[str, float]], str]:
    """Send one image to Colab and return ranked labels plus its device name."""
    if not api_url.strip() or not api_token:
        raise ValueError("Colab GPU inference requires an API URL and token")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    response = requests.post(
        f"{api_url.rstrip('/')}/predict",
        headers={"X-API-Key": api_token},
        files={"file": (filename, image_bytes, content_type)},
        data={"top_k": str(top_k)},
        timeout=timeout,
    )
    response.raise_for_status()
    payload: Any = response.json()
    raw_predictions = payload.get("predictions") if isinstance(payload, dict) else None
    if not isinstance(raw_predictions, list) or not raw_predictions:
        raise ValueError("Colab returned an invalid prediction response")

    predictions = [
        (str(item["class_name"]), float(item["confidence"]))
        for item in raw_predictions
    ]
    return predictions, str(payload.get("device", "Colab GPU"))