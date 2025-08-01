from flask import Flask
from flask_restx import Api
from flask_cors import CORS
# from diffusers import DiffusionPipeline
# import torch
# import io
# from rembg import remove
from .ml_logic import load_models
from .api import ns_images

def create_app():
    """Creates and configures the Flask app"""
    app = Flask(__name__)
    CORS(app)

    # Initialise the API
    api = Api(app, validate=True, version='1.0', title='Diffusion API', description='Provides a Stable Diffusion generated Image based on the provided prompt.')

    # Add the Images Namespace
    api.add_namespace(ns_images)

    # Load models and attach them to the app instance
    with app.app_context():
        app.base_model, app.refiner_model = load_models()
    
    return app

# This block is for local debugging and is not used by Gunicorn
if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)

# Command to run locally: python3 -m gunicorn --bind 127.0.0.1:5000 -w 1 --timeout 300 "src.main:create_app()"