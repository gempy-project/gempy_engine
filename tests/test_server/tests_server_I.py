import json
import os

import pytest
from fastapi.testclient import TestClient

from gempy_engine.API.server.main_server_pro import gempy_engine_App

# Define paths relative to the project root

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_JSON_PATH = os.path.join(BASE_PATH, "example.json")
FEATURES_JSON_PATH = os.path.join(BASE_PATH, "2features.json")


def load_request_data(json_file_path):
    """Load JSON data from a file path"""
    try:
        with open(json_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: File '{json_file_path}' not found") from e
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error: File '{json_file_path}' contains invalid JSON", e.doc, e.pos) from e


@pytest.fixture
def client():
    """Create a test client for FastAPI app"""
    return TestClient(gempy_engine_App)


def test_post_example_json(client):
    """Test POST request with example.json data"""
    data = load_request_data(EXAMPLE_JSON_PATH)
    assert data is not None, f"Failed to load data from {EXAMPLE_JSON_PATH}"
    
    response = client.post(
        "/",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    assert len(response.content) == 15326
    assert response.status_code == 200
    
    # Save binary into example.mle
    with open("example.mle", "wb") as f:
        f.write(response.content)


@pytest.mark.skip(reason="Not implemented yet")
def test_post_features_json(client):
    raise NotImplementedError
    """Test POST request with 2features.json data"""
    data = load_request_data(FEATURES_JSON_PATH)
    assert data is not None, f"Failed to load data from {FEATURES_JSON_PATH}"
    
    response = client.post(
        "/",
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "result" in result or "scalar_field" in result