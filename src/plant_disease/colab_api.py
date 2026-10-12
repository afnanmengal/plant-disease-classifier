"""Authenticated FastAPI inference service for a Colab GPU runtime."""

import hmac
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

from plant_disease.model import load_checkpoint, predict_image_top_k

MODEL = None
CLASS_NAMES: list[str] = []
IMAGE_SIZE = 224
DEVICE_NAME = ""


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global CLASS_NAMES, DEVICE_NAME, IMAGE_SIZE, MODEL

    if not torch.cuda.is_available():
        raise RuntimeError("This Colab inference service requires an active CUDA GPU runtime")
    checkpoint_path = Path(
        os.environ.get("PLANT_DISEASE_CHECKPOINT", "models/plant_disease_resnet18.pt")
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    MODEL, CLASS_NAMES, IMAGE_SIZE = load_checkpoint(checkpoint_path, device="cuda")
    DEVICE_NAME = torch.cuda.get_device_name(0)
    yield


app = FastAPI(title="Plant Disease Colab GPU", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"status": "ready", "device": DEVICE_NAME}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    top_k: int = Form(default=3, ge=1, le=5),
    api_key: str = Header(default="", alias="X-API-Key"),
) -> dict[str, object]:
    expected_key = os.environ.get("PLANT_DISEASE_API_KEY", "")
    if not expected_key or not hmac.compare_digest(api_key, expected_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    image_bytes = await file.read(12 * 1024 * 1024 + 1)
    if not image_bytes or len(image_bytes) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be between 1 byte and 12 MB")
    try:
        predictions = predict_image_top_k(
            MODEL,
            io.BytesIO(image_bytes),
            CLASS_NAMES,
            image_size=IMAGE_SIZE,
            device="cuda",
            top_k=top_k,
        )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Unable to decode image") from error
    return {
        "device": DEVICE_NAME,
        "predictions": [
            {"class_name": class_name, "confidence": confidence}
            for class_name, confidence in predictions
        ],
    }