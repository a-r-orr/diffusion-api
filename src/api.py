from flask import send_file, current_app
from flask_restx import Resource, fields, Namespace
import io
import torch

from .ml_logic import create_image

# Create Images Namespace
ns_images = Namespace('images', description='Image operations')


# Model for expected prompt
prompt_model = ns_images.model('Prompt', {
    'prompt': fields.String(required=True, min_length=1, description='The text prompt to generate the image.')
})

# /create-image route
@ns_images.route('/create-image')
class GenImage(Resource):
    @ns_images.doc('create_image_from_prompt')
    @ns_images.expect(prompt_model)
    @ns_images.produces(['image/png'])
    def post(self):
        '''Route for creating a new image based on a prompt'''
        prompt = ns_images.payload['prompt']
        adj_prompt = "A 3D model of " + prompt + " with a blank background"

        # Create the image
        image = create_image(adj_prompt, current_app.base_model, current_app.refiner_model)

        if image is None:
            return {'message': 'Image generation failed on the server.'}, 500

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        torch.cuda.empty_cache()
        return send_file(buffer, mimetype='image/png')