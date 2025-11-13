# tests/test_api.py

import pytest
from fastapi.testclient import TestClient
from scripts.rag_fast_api import app

client = TestClient(app)

def test_read_root():
    """Test de la route racine"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_search_endpoint():
    """Test de l'endpoint de recherche"""
    response = client.get("/search?query=concert&k=3")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert isinstance(data["results"], list)

def test_ask_endpoint():
    """Test de l'endpoint de question/réponse"""
    response = client.post("/ask", json={"question": "Quels événements à Paris ?", "k": 2})
    assert response.status_code == 200
    data = response.json()
    assert "question" in data
    assert "generated_answer" in data
'''
def test_health_check():
    """Test de l'endpoint health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"








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
'''    