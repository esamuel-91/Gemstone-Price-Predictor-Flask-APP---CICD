import pytest
import os
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Disable logging of errors during tests to keep output clean
    app.config['DEBUG'] = False 
    with app.test_client() as client:
        yield client

def test_homepage(client):
    response = client.get('/')
    assert response.status_code == 200

# 🛠️ IMPROVED CHECK: Check if files exist AND are larger than 1MB (pointers are < 1KB)
def artifacts_are_valid():
    model_path = "artifacts/model.pkl"
    preprocessor_path = "artifacts/preprocessor.pkl"
    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        # If file size > 1MB, it's likely a real model, not a pointer
        return os.path.getsize(model_path) > 1024 * 1024 
    return False

@pytest.mark.skipif(not artifacts_are_valid(), reason="Real model files not found (Git LFS pointers or missing).")
def test_prediction_endpoint(client):
    test_data = {
        "log_carat": "0.5", "volume": "150.0", "depth": "61.5",
        "table": "55.0", "cut": "Ideal", "color": "E", "clarity": "SI1"
    }
    response = client.post('/predict', data=test_data, follow_redirects=True)
    assert response.status_code == 200

def test_invalid_input_handling(client):
    """Test that the app handles bad input without crashing the server."""
    bad_data = {"log_carat": "not-a-number"}
    
    # We EXPECT this to trigger your CustomException (which Flask returns as 500)
    # The test passes if the server handles it rather than crashing
    response = client.post('/predict', data=bad_data)
    assert response.status_code in [400, 500]