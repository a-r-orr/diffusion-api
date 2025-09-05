# Diffusion API
This Flask API returns an image generated with Stable Diffusion, based on a text prompt.

## Stable Diffusion
Stability AI's "stable-diffusion-xl-base-1.0" model that is used can be found on Hugging Face: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

## Installation
This has only been tested in a Linux dev environment, using Python 3.12.3.

Other Operating Systems may have issues with running the ML models.

### Local Environment
To run the API locally, I recommend creating a Conda environment to manage installed packages and dependencies.
```
conda create --name <my-env>
conda activate <my-env>
conda install pip
```
Replace ```<my-env>``` with the name of your environment.
More information can be found here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### PyTorch
PyTorch should be installed first.
Follow instructions for your system here: https://pytorch.org/get-started/locally/

e.g. for my system running Linux with CUDA 12.8:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Requirements
Once PyTorch is installed, then install the remaining requirements:
```
pip install -r requirements.txt
```

## Usage
The API can be run locally (currently set to PORT 5000 in the last line of main.py):
```
python3 -m gunicorn --bind 127.0.0.1:5000 -w 1 --timeout 300 "src.main:create_app()"
```
Whatever you are using to consume the API (e.g. Postman) should send POST requests to the "/images/create-image" endpoint, with the prompt contained in the body:
```
{
    "prompt": "an emu riding a motorcycle"
}
```

A Dockerfile has been provided to enable containerization of the API for deployment.
With the current setup, running the below docker command will host the API locally on port 5000 once you have containerized the project:
```
docker run --rm -it   --gpus all   -p 5000:8080   -e PORT=8080   -v "$(pwd)/local_model_cache:/home/appuser/.cache"   diffusion-api
```

## Licence
My code is free to use, but please refer to the licencing details provided on Stability AI's Hugging Face page.