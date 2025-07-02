from flask import Flask, send_file
from flask_restx import Api, Resource, fields, Namespace
from diffusers import DiffusionPipeline
import torch
import io
from rembg import remove

base = None
refiner = None

def get_models():
    """Loads and returns the models, loading only if they haven't been loaded yet."""
    global base, refiner
    
    if base is None or refiner is None:
        print("Loading models for the first time...")
        # Load base model
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True, 
            cache_dir="./local_model_cache"
        )
        # Enable CPU offloading
        base.enable_model_cpu_offload()

        # Load refiner model
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir="./local_model_cache"
        )
        # Enable CPU offloading
        refiner.enable_model_cpu_offload()
        print("Models loaded successfully.")

    return base, refiner


def create_image(prompt, base, refiner):
    # run both experts

    # Define how many steps and what % of steps to be run on each experts (80/20) here
    n_steps = 40
    high_noise_frac = 0.8

    torch.cuda.empty_cache()
    image = base(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    torch.cuda.empty_cache()
    return image


app = Flask(__name__)
api = Api(app, version='1.0', title='Diffusion API', description='Provides a Stable Diffusion generated Image based on the provided prompt.')

# Create Images Namespace
ns_images = Namespace('images', description='Image operations')
api.add_namespace(ns_images)

# Model for expected prompt
prompt_model = ns_images.model('Prompt', {
    'prompt': fields.String(required=True, description='The text prompt to generate the image.')
})

# /create-image route
@ns_images.route('/create-image')
class GenImage(Resource):
    
    @ns_images.doc('create_image_from_prompt')
    @ns_images.expect(prompt_model)
    @ns_images.produces(['image/png'])

    def post(self):
        '''Route for creating a new image based on a prompt'''
        base, refiner = get_models()

        prompt = api.payload['prompt']
        image = create_image(prompt, base, refiner)

        output = remove(image)

        buffer = io.BytesIO()
        output.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')

if __name__ == "__main__":
     app.run(debug=True)