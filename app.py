"""Streamlit interface for classifying a leaf image."""

import hashlib
import io
import os
import sys
from pathlib import Path

import requests
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from plant_disease.colab_client import predict_with_colab
from plant_disease.model import load_checkpoint, predict_image_top_k


st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root {
        color-scheme: light;
        --night: #f4f8f4;
        --panel: #ffffff;
        --panel-raised: #edf5ef;
        --text: #15291f;
        --muted: #536c5d;
        --line: rgba(47, 104, 72, 0.2);
        --mint: #197452;
        --lime: #91be42;
        --cyan: #18777b;
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--night);
        background-image:
            linear-gradient(rgba(73, 137, 93, 0.075) 1px, transparent 1px),
            linear-gradient(90deg, rgba(73, 137, 93, 0.075) 1px, transparent 1px),
            radial-gradient(ellipse at 48% -18%, rgba(194, 224, 199, 0.48), transparent 58%);
        background-size: 42px 42px, 42px 42px, 100% 100%;
        color: var(--text);
    }
    [data-testid="stHeader"] { background: rgba(248, 251, 248, 0.94); }
    [data-testid="stSidebar"] {
        background: rgba(237, 245, 239, 0.97);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div { background: transparent; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { color: var(--text) !important; }
    .block-container { max-width: 1440px; padding: 2.8rem 2.7rem 2rem; }
    h1, h2, h3, p, label { color: var(--text); }
    h1, h2, h3 { font-family: 'Space Grotesk', 'Avenir Next', sans-serif; letter-spacing: 0; }
    h1 { font-size: 3rem; font-weight: 600; line-height: 1.08; }
    h2 { font-size: 1.55rem; font-weight: 500; }
    h3 { font-size: 1rem; font-weight: 500; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--muted); }
    [data-testid="stMetric"] {
        background: linear-gradient(110deg, #ffffff, #edf5ef);
        border: 1px solid var(--line);
        border-left: 3px solid var(--mint);
        padding: 0.85rem 1rem;
        border-radius: 5px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-family: 'DM Mono', monospace; }
    [data-testid="stMetricValue"] { color: #27633e; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stFileUploader"] section {
        min-height: 132px;
        background: rgba(251, 253, 251, 0.94);
        border: 1px dashed rgba(43, 126, 78, 0.62);
        border-radius: 7px;
    }
    [data-testid="stFileUploader"] section:hover {
        background: #edf6ef;
        border-color: #6ca54d;
    }
    [data-testid="stFileUploader"] section p,
    [data-testid="stFileUploader"] section span,
    [data-testid="stFileUploader"] small { color: var(--muted) !important; }
    [data-testid="stFileUploader"] button,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: var(--panel-raised);
        border-color: var(--line);
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-baseweb="select"] * { color: var(--text); }
    [data-testid="stButton"] button[kind="primary"] {
        background: var(--lime);
        color: #102016;
        border: 1px solid var(--lime);
        border-radius: 4px;
        font-family: 'DM Mono', monospace;
        font-weight: 500;
        min-height: 46px;
        transition: box-shadow 160ms ease, transform 160ms ease;
    }
    [data-testid="stButton"] button[kind="primary"]:hover {
        box-shadow: 0 0 18px rgba(99, 151, 57, 0.24);
        transform: translateY(-1px);
    }
    [data-testid="stProgressBar"] > div > div > div > div {
        background: linear-gradient(90deg, var(--mint), var(--lime));
    }
    [data-testid="stAlert"] {
        background: rgba(236, 245, 238, 0.96);
        border: 1px solid var(--line);
        color: var(--text);
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(226, 240, 230, 0.98);
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] p { color: var(--text); }
    hr { border-color: var(--line); }
    code, [data-testid="stCode"] {
        background: #f0f6f1;
        border: 1px solid var(--line);
        color: var(--mint);
    }
    .system-rail {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0 0 1rem;
        color: var(--mint);
        font: 500 0.72rem 'DM Mono', monospace;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .system-rail-left { display: flex; align-items: center; gap: 0.65rem; }
    .signal-mark {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--lime);
        box-shadow: 0 0 12px rgba(200, 255, 107, 0.8);
        animation: signal-pulse 2.2s ease-in-out infinite;
    }
    .system-code { color: var(--muted); }
    @keyframes signal-pulse {
        50% { box-shadow: 0 0 3px rgba(200, 255, 107, 0.25); opacity: 0.65; }
    }
    @media (max-width: 700px) {
        .block-container { padding: 1.4rem 1rem 1.3rem; }
        h1 { font-size: 2.15rem; }
        .system-rail { align-items: flex-start; flex-direction: column; gap: 0.4rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_cached_model(checkpoint: str, modified_ns: int, device: str):
    return load_checkpoint(checkpoint, device=device)


def get_setting(secret_name: str, environment_name: str) -> str:
    try:
        secret_value = st.secrets.get(secret_name, "")
    except (FileNotFoundError, RuntimeError):
        secret_value = ""
    return str(secret_value or os.environ.get(environment_name, "")).strip()


st.markdown(
    '<div class="system-rail"><span class="system-rail-left">'
    '<span class="signal-mark"></span> Botanical signal analysis</span>'
    '<span class="system-code">PLANTVILLAGE / NODE 01 / RESNET18</span></div>',
    unsafe_allow_html=True,
)
st.title("Leaf health scan")
st.caption("Plant disease recognition | 15-class field image model")
st.divider()

checkpoint_default = PROJECT_ROOT / "models" / "plant_disease_resnet18.pt"
checkpoint_path = Path(st.sidebar.text_input("Model checkpoint", str(checkpoint_default)))

colab_setting_url = get_setting("colab_api_url", "PLANT_DISEASE_COLAB_API_URL")
colab_setting_token = get_setting("colab_api_token", "PLANT_DISEASE_COLAB_API_TOKEN")

with st.sidebar.expander("Colab GPU Endpoint Settings", expanded=bool(colab_setting_url)):
    colab_api_url = st.text_input(
        "Colab API URL",
        value=colab_setting_url,
        placeholder="https://...trycloudflare.com",
        help="Temporary tunnel URL from Google Colab",
    ).strip()
    colab_api_token = st.text_input(
        "Colab API Token",
        value=colab_setting_token,
        type="password",
        placeholder="Secret API key",
        help="Private token matching PLANT_DISEASE_API_KEY in Colab",
    ).strip()

colab_configured = bool(colab_api_url and colab_api_token)
st.sidebar.subheader("Inference device")
cuda_available = torch.cuda.is_available()
cuda_name = torch.cuda.get_device_name(0) if cuda_available else ""
cuda_compatible = False
if cuda_available:
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    cuda_compatible = architecture in torch.cuda.get_arch_list()
    if cuda_compatible:
        st.sidebar.success(f"NVIDIA GPU ready: {cuda_name}")
    else:
        st.sidebar.warning(
            f"{cuda_name} detected, but this PyTorch build lacks {architecture} support. "
            "Install the CUDA 12.8 dependencies to use it."
        )
else:
    st.sidebar.info("No CUDA-capable NVIDIA GPU detected; inference will use CPU.")

if colab_configured:
    st.sidebar.success("Colab GPU endpoint configured")
elif colab_api_url or colab_api_token:
    st.sidebar.warning("Set both Colab API URL and token to enable remote GPU inference.")

device_labels = ["Colab GPU"] if colab_configured else []
if cuda_compatible:
    device_labels.append("NVIDIA GPU")
device_labels.append("CPU")
selected_device = st.sidebar.selectbox(
    "Run inference on", device_labels, index=0
)
device = "cuda" if selected_device == "NVIDIA GPU" else "cpu"
uploaded_image = st.file_uploader(
    "Choose a leaf photo", type=["jpg", "jpeg", "png"], help="Use a clear, well-lit leaf image."
)

image_column, result_column = st.columns([1.05, 0.95], gap="large")
with image_column:
    st.subheader("Image")
    image_bytes = uploaded_image.getvalue() if uploaded_image is not None else None
    image_preview = None
    if image_bytes:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_preview = image.convert("RGB")
        except (OSError, ValueError):
            st.error("This file is not a readable image. Try a different JPG or PNG.")
        if image_preview is not None:
            st.image(image_preview, use_container_width=True)
    else:
        st.info("Upload a JPG or PNG image to begin a scan.")

with result_column:
    st.subheader("Classification")
    if not checkpoint_path.is_file() and selected_device != "Colab GPU":
        st.warning("Model checkpoint not found. Train the classifier to enable scans.")
        st.code(
            "python scripts/train.py --device cuda `\n"
            "  --data-dir \"PATH_FROM_DOWNLOAD_OUTPUT\"",
            language="powershell",
        )
    elif not image_bytes:
        st.info("Your results will appear here after you add an image.")
    elif image_preview is None:
        st.info("Upload a readable leaf image to begin a scan.")
    elif device == "cuda" and not cuda_compatible:
        st.error(
            "The installed PyTorch build cannot run this GPU. Choose CPU or update dependencies."
        )
    elif selected_device == "Colab GPU" and len(image_bytes) > 12 * 1024 * 1024:
        st.error("Colab's shared GPU service accepts images up to 12 MB.")
    elif st.button("Analyze leaf", type="primary", use_container_width=True):
        try:
            fingerprint = hashlib.sha256(image_bytes).hexdigest()
            with st.spinner("Analyzing leaf image..."):
                if selected_device == "Colab GPU":
                    predictions, inference_device = predict_with_colab(
                        colab_api_url,
                        colab_api_token,
                        image_bytes,
                        uploaded_image.name,
                        uploaded_image.type or "application/octet-stream",
                    )
                    checkpoint_modified = 0
                else:
                    checkpoint_modified = checkpoint_path.stat().st_mtime_ns
                    model, class_names, image_size = get_cached_model(
                        str(checkpoint_path), checkpoint_modified, device
                    )
                    predictions = predict_image_top_k(
                        model,
                        io.BytesIO(image_bytes),
                        class_names,
                        image_size=image_size,
                        device=device,
                        top_k=3,
                    )
                    inference_device = selected_device
            st.session_state["leaf_scan"] = {
                "fingerprint": fingerprint,
                "checkpoint": str(checkpoint_path),
                "checkpoint_modified": checkpoint_modified,
                "device": inference_device,
                "route": selected_device,
                "predictions": predictions,
            }
        except (OSError, KeyError, RuntimeError, ValueError, requests.RequestException) as error:
            st.error(f"Could not classify this image: {error}")

    scan = st.session_state.get("leaf_scan")
    if (
        image_preview is not None
        and scan
        and scan["fingerprint"] == hashlib.sha256(image_bytes).hexdigest()
        and scan["checkpoint"] == str(checkpoint_path)
        and scan["checkpoint_modified"]
        == (0 if selected_device == "Colab GPU" else checkpoint_path.stat().st_mtime_ns)
        and scan["route"] == selected_device
    ):
        best_name, best_score = scan["predictions"][0]
        st.metric("Most likely class", best_name.replace("___", " - ").replace("_", " "))
        st.metric("Model confidence", f"{best_score:.1%}")
        st.caption(f"Inference device: {scan['device']}")
        st.markdown("**Other likely classes**")
        for class_name, score in scan["predictions"][1:]:
            label = class_name.replace("___", " - ").replace("_", " ")
            st.progress(score, text=f"{label} | {score:.1%}")

st.divider()
st.caption(
    "Screening aid only, not an agricultural diagnosis. PlantVillage photos use controlled "
    "backgrounds, so field-photo accuracy may be lower. Verify suspected disease with an "
    "agricultural specialist. Model confidence is not a calibrated probability."
)