# Plant Disease Classifier

Train an image classifier for common plant diseases using transfer learning with an ImageNet-pretrained ResNet18. The first dataset target is PlantVillage, a labeled collection organized by crop and disease class. The project includes reproducible train/validation splitting, a training CLI, and a Streamlit image-prediction app.

This is a learning and screening aid, not an agricultural diagnosis. PlantVillage images are controlled-background photographs; performance on field photos can be substantially lower. Verify suspected disease with a qualified agricultural specialist.

## Setup

Create the local NVIDIA environment from the project folder:

```powershell
conda env create -f environment.yml
conda activate plant-disease-classifier
```

In `notebooks/01_data_processing.ipynb`, select the `Plant Disease Classifier (CUDA)` Python kernel so notebook processing and CUDA checks use the same environment as training.

The Conda environment uses `requirements-gpu.txt`, which selects PyTorch's CUDA 12.8 wheels. Training requires CUDA by default and checks that the installed PyTorch build supports the GPU architecture. Use `--device cpu` only when you intentionally want CPU training.

`requirements.txt` is the CPU-compatible runtime set used by Streamlit hosting. The NVIDIA driver must be installed for local GPU work; a separate CUDA Toolkit is not required because the PyTorch wheel includes its CUDA runtime.

## Download and Train

Download the public dataset with KaggleHub:

```powershell
python scripts/download_dataset.py
```

The script prints the local directory whose immediate subfolders are disease classes. Pass that exact path to training:

```powershell
python scripts/train.py --data-dir "PATH_FROM_DOWNLOAD_OUTPUT" --device cuda
```

Training downloads pretrained ResNet18 weights on its first run, then saves the best validation-accuracy checkpoint to `models/plant_disease_resnet18.pt`. KaggleHub may require Kaggle account credentials depending on your environment. For a CPU smoke run, pass `--device cpu --epochs 1 --batch-size 8 --no-pretrained`.

Useful options include `--epochs`, `--batch-size`, `--learning-rate`, `--image-size`, `--num-workers`, `--output`, and `--device cuda|cpu`.

## Google Colab GPU

Open `notebooks/02_colab_gpu.ipynb` in Google Colab and select a GPU runtime. Run the notebook to clone this public repository, download PlantVillage, train with Colab's GPU, and copy the checkpoint to Google Drive. Colab's existing PyTorch installation is reused.

The notebook can also start an authenticated FastAPI inference service and a temporary Cloudflare tunnel. In Colab's Secrets panel, create `PLANT_DISEASE_API_KEY`. In Streamlit Community Cloud, add the tunnel URL printed by the notebook and the same private token under **App settings > Secrets**:

```toml
colab_api_url = "https://your-temporary-tunnel.trycloudflare.com"
colab_api_token = "your-private-token"
```

When both secrets are set, the app offers **Colab GPU** as its default inference route. The Colab notebook must remain connected; its URL changes when the runtime restarts, so update `colab_api_url` after reconnecting. Without Colab secrets, the app uses its local NVIDIA GPU when available, otherwise CPU.

For Streamlit Community Cloud, deploy `app.py` from this repository. It installs the CPU-compatible `requirements.txt`; do not point it at `requirements-gpu.txt` because hosted Streamlit runtimes do not provide a CUDA GPU.

## Run the App and Checks

```powershell
streamlit run app.py
python -m pytest
python -m ruff check .
```

Use the app's sidebar to select a checkpoint, then upload a JPG or PNG leaf photo. The default checkpoint is `models/plant_disease_resnet18.pt`.

## Project Layout

- `src/plant_disease/data.py`: ImageFolder loading, stratified split, and image transforms.
- `src/plant_disease/model.py`: ResNet18 setup, checkpoint loading, and image inference.
- `src/plant_disease/training.py`: training and best-checkpoint saving.
- `src/plant_disease/colab_api.py`: token-protected inference service for a live Colab GPU.
- `scripts/`: dataset download and training commands.
- `app.py`: Streamlit prediction interface.
- `notebooks/02_colab_gpu.ipynb`: Colab training, Drive backup, and optional temporary GPU API.
- `tests/`: lightweight model and data-input checks.

Image data and checkpoints are local artifacts and are excluded from Git.