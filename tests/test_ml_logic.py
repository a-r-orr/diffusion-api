import pytest
from unittest.mock import MagicMock
from src.ml_logic import create_image

def test_create_image_pipeline_logic():
    """
    Unit tests the create_image function to ensure the pipeline
    calls the base and refiner models with the correct parameters.
    """
    # Create mock models
    mock_base = MagicMock()
    mock_refiner = MagicMock()
    
    # Configure the mocks to return a value in the expected structure
    # to avoid errors on chained calls like .images[0]
    mock_base.return_value.images = ["latent_image_output"]
    mock_refiner.return_value.images = ["final_image_output"]

    prompt = "a test prompt"
    n_steps = 40
    high_noise_frac = 0.8

    # Call the function with the mock models
    result = create_image(prompt, mock_base, mock_refiner)

    # Check that the final result is what the refiner returned
    assert result == "final_image_output"

    # Check that the base model was called correctly
    mock_base.assert_called_once_with(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    )

    # Check that the refiner model was called correctly
    mock_refiner.assert_called_once_with(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=["latent_image_output"],
    )