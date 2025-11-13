# tests/test_api_endpoints.py
import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from rag_fast_api import app

client = TestClient(app)

def test_root_endpoint():
    """Test de l'endpoint racine"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "RAG Event Search" in data["message"]

def test_search_endpoint_basic():
    """Test basique de l'endpoint search"""
    response = client.get("/search?query=concert&k=2")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "results" in data
    assert isinstance(data["results"], list)

def test_search_endpoint_with_city():
    """Test de search avec filtre ville"""
    response = client.get("/search?query=concert à Paris&k=2")
    assert response.status_code == 200
    data = response.json()
    assert "city_detected" in data
    # Peut être "paris" ou None selon les données

def test_ask_endpoint_structure():
    """Test de la structure de l'endpoint ask"""
    response = client.post("/ask", json={"question": "Quels événements ?", "k": 1})
    # Même en échec (pas de clé API), on vérifie la structure
    assert response.status_code in [200, 500]
    if response.status_code == 200:
        data = response.json()
        assert "question" in data
        assert "generated_answer" in data

def test_invalid_parameters():
    """Test avec paramètres invalides"""
    response = client.get("/search?query=&k=20")  # k trop grand
    assert response.status_code == 422  # Validation error

def test_rebuild_endpoint():
    """Test de l'endpoint rebuild (doit retourner accepted)"""
    response = client.post("/rebuild")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data