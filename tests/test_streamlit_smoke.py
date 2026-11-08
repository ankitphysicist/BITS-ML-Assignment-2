"""Smoke-test the Streamlit app against saved model artifacts."""
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]

def test_streamlit_app_starts_and_scores_saved_models():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45)
    app.run()
    assert len(app.exception) == 0, [str(e.message) for e in app.exception]
    assert len(app.metric) >= 9
    assert len(app.selectbox) >= 1
