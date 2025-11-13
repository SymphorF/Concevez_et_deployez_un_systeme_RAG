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

def test_health_check():
    """Test de l'endpoint health"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"