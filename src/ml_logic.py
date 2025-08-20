from diffusers import DiffusionPipeline
import torch

# Function to load the diffusion models
def load_models():
    """Loads and returns the models from the local cache."""
    print("Loading models from local cache...")

    # Load base model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # Load refiner model
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )

    # CPU offload enabled for both models to minimise GPU usage when not actively generating
    # ---- These Lines enable CPU Offload - comment out if running on dedicated GPU ---
    base.enable_model_cpu_offload()
    refiner.enable_model_cpu_offload()
    # --- --- --- --- --- --- ---
    
    # ---- These lines move the models to the GPU - comment out if using CPU Offloading ---
    # base.to("cuda")
    # refiner.to("cuda")
    # --- --- --- --- --- --- ---

    print("Models loaded successfully and ready.")
    return base, refiner

# Function to create image using the base and refiner models
def create_image(prompt, base, refiner):
    """Creates and image based on provided prompt, using the base and refiner models provided."""
    try:
        # Run both models
        # These parameters control how many steps and the ratio between models
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
    except Exception as e:
        print(f"Image generation failed: {e}")
        torch.cuda.empty_cache()
        return None