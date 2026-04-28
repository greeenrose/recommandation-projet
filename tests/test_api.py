import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_list_students():
    response = client.get("/students")
    assert response.status_code == 200
    assert "students" in response.json()

def test_recommend_valid_user():
    response = client.get("/recommend/1")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0

def test_recommend_invalid_user():
    response = client.get("/recommend/9999")
    assert response.status_code == 404