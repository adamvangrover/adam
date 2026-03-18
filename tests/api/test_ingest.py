from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

def test_health_check():
    """
    Test the basic health endpoint
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_upload_no_file():
    """
    Test that uploading without a file results in a 422 Unprocessable Entity
    """
    response = client.post("/api/v1/ingest/upload", data={"instructions": "Test", "model_name": "gpt-4o"})
    assert response.status_code == 422 # Pydantic validation error for missing field
