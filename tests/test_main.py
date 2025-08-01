import pytest
from unittest.mock import MagicMock
from PIL import Image
import io

from src.main import create_app

# Create fixture for test client
@pytest.fixture
def client(mocker):
    mocker.patch('src.main.load_models', return_value=(MagicMock(), MagicMock()))
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# Fixture to create dummy image for mocking
@pytest.fixture
def dummy_image():
    """Creates a simple PIL Image for testing."""
    img = Image.new('RGB', (100,100), color = 'red')
    return img

# Fixture for failed image generation for mocking
@pytest.fixture
def failed_image():
    """Returns None to simulate the image generation having failed."""
    return None

def test_create_image_endpoint(client, mocker, dummy_image):

    # Mock the main.create_image function to enable faster testing
    mock_create = mocker.patch('src.api.create_image', return_value=dummy_image)

    # Simulate request to the endpoint
    prompt_text = "an emu in sunglasses"
    response = client.post('/images/create-image', json={'prompt': prompt_text})

    # Check the response
    assert response.status_code == 200
    assert response.mimetype == 'image/png'

    # Check that the mock was called correctly
    expected_prompt = "A 3D model of " + prompt_text + " with a blank background"
    mock_create.assert_called_once()
    mock_create.assert_called_with(expected_prompt, mocker.ANY, mocker.ANY)

    # Check that response data contains the dummy image
    response_data = io.BytesIO(response.data)
    returned_image = Image.open(response_data)
    assert returned_image.format == 'PNG'
    assert returned_image.size == (100,100)

def test_failure_create_image_endpoint(client, mocker, failed_image):
    
    # Mock the main.create_image function to enable faster testing
    mock_create = mocker.patch('src.api.create_image', return_value=failed_image)

    # Simulate request to the endpoint
    prompt_text = "an emu in sunglasses"
    response = client.post('/images/create-image', json={'prompt': prompt_text})

    # Check the response
    assert response.status_code == 500
    assert response.json == {"message": "Image generation failed on the server."}

def test_missing_prompt(client):
    """Test that an empty request body returns a 400 Bad Request"""
    response = client.post('/images/create-image', json={})

    assert response.status_code == 400
    assert response.json.get('message') == "Input payload validation failed"

def test_invalid_prompt(client):
    """Test that a non-string prompt returns a 400 Bad Request"""
    response = client.post('/images/create-image', json={'prompt': 999})

    assert response.status_code == 400
    assert response.json.get('message') == "Input payload validation failed"

def test_empty_prompt(client):
    """Test that a prompt containing an empty string returns a 400 Bad Request"""
    response = client.post('/images/create-image', json={'prompt': ''})

    assert response.status_code == 400
    assert response.json.get('message') == "Input payload validation failed"


# Finally, spin up a version of the client without any mocking.
# This will allow to test the ML functionality and confirm that an image is returned.
@pytest.fixture
def full_client():
    """Fixture to provide a full client without any mocking."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.mark.e2e
def test_e2e_image_generation(full_client):
    """
    End-to-end test - tests the actual operation of the image generation.
    This test is slow and needs a GPU.
    """
    prompt_text = "an emu in sunglasses"
    response = full_client.post('/images/create-image', json={'prompt': prompt_text})

    # Check the response
    assert response.status_code == 200
    assert response.mimetype == 'image/png'

    try:
        response_data = io.BytesIO(response.data)
        with Image.open(response_data) as img:
            assert img.format == 'PNG'
            assert img.width > 0
            assert img.height > 0
    except Exception as e:
        pytest.fail(f"Error while trying to open the response data: {e}")