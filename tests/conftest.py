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

# Fixture with fully functional client for E2E test
@pytest.fixture
def full_client():
    """Fixture to provide a full client without any mocking."""
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client