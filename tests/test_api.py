import pytest
import sys
import os

# Configuration du path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def test_imports():
    """Test que les imports fonctionnent"""
    try:
        from rag_fast_api import app
        from app_config import MISTRAL_API_KEY, OPENAGENDA_API_KEY
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

def test_app_creation():
    """Test que l'app FastAPI se crée correctement"""
    from rag_fast_api import app
    assert app.title == "RAG Event Search API"
    assert app.version == "1.0" 