# tests/test_api_no_embeddings.py
import pytest
import sys
import os
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

def test_api_structure_only():
    """Test uniquement la structure de l'API sans lancer les embeddings"""
    from rag_fast_api import app
    
    client = TestClient(app)
    
    # Test des endpoints qui ne nécessitent pas d'embeddings
    response = client.get("/")
    assert response.status_code == 200
    
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    
    response = client.post("/rebuild")
    assert response.status_code == 200

def test_search_with_empty_query():
    """Test de search avec query vide (devrait échouer en validation)"""
    from rag_fast_api import app
    client = TestClient(app)
    
    # Test avec query vide
    response = client.get("/search?query=&k=1")
    assert response.status_code == 422  # Validation error
    assert "detail" in response.json()

def test_search_with_spaces_only():
    """Test de search avec uniquement des espaces"""
    from rag_fast_api import app
    client = TestClient(app)
    
    response = client.get("/search?query=%20%20%20&k=1")
    assert response.status_code == 422  # Validation error

def test_search_with_valid_query():
    """Test de search avec une requête valide"""
    from rag_fast_api import app
    client = TestClient(app)
    
    response = client.get("/search?query=test&k=3")
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["query"] == "test"
    assert data["k"] == 3

def test_search_with_invalid_k():
    """Test de search avec une valeur de k invalide"""
    from rag_fast_api import app
    client = TestClient(app)
    
    response = client.get("/search?query=test&k=0")
    assert response.status_code == 422  # Validation error

def test_search_with_large_k():
    """Test de search avec k trop grand"""
    from rag_fast_api import app
    client = TestClient(app)
    
    response = client.get("/search?query=test&k=25")
    assert response.status_code == 422  # Validation error