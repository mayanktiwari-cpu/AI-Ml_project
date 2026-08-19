from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/docs")
    assert response.status_code == 200