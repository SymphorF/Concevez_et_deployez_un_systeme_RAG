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
    
    response = client.post("/rebuild")
    assert response.status_code == 200

def test_search_with_empty_query():
    """Test de search avec query vide (devrait échouer en validation)"""
    from rag_fast_api import app
    client = TestClient(app)
    
    response = client.get("/search?query=&k=1")
    assert response.status_code == 422  # Validation error