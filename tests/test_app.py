from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_renders_without_a_checkpoint():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert [heading.value for heading in app.title] == ["Leaf health scan"]


def test_app_offers_colab_gpu_when_api_settings_are_present(monkeypatch):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    monkeypatch.setenv("PLANT_DISEASE_COLAB_API_URL", "https://colab.example")
    monkeypatch.setenv("PLANT_DISEASE_COLAB_API_TOKEN", "test-token")

    app = AppTest.from_file(str(app_path)).run(timeout=30)

    assert not app.exception
    assert "Colab GPU" in app.selectbox[0].options