import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import cv2
import numpy as np

def generate_controlled_image(prompt: str, reference_image_path: str):
    """
    Demonstrates programmatic image generation guided by ControlNet structures.
    Requires significant VRAM (12GB+ recommended).
    """
    # 1. Process reference image for Edge Detection
    image = load_image(reference_image_path)
    image = np.array(image)
    low_threshold, high_threshold = 100, 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = load_image(image)

    # 2. Load ControlNet and SD BaseModel in fp16
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    
    # 3. Optimize Scheduling and VRAM
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # 4. Generate
    output = pipe(
        prompt,
        image=canny_image,
        num_inference_steps=20,
        guidance_scale=7.5
    ).images[0]
    
    output.save("vibe_generated_output.png")

if __name__ == "__main__":
    # Example execution:
    # generate_controlled_image("highly detailed cyberpunk city, neon lights", "assets/input.jpg")
    pass
