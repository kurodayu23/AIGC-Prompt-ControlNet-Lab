from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image

from prompt_engine import build_controlnet_prompt


def generate_controlled_image(subject: str, scene: str, style: str, reference_image_path: str, output_path: str = "output.png") -> str:
    """
    Programmatic image generation guided by ControlNet (Canny).

    Returns output image path for downstream automation.
    """
    prompt = build_controlnet_prompt(subject=subject, scene=scene, style=style)

    image = load_image(reference_image_path)
    image = np.array(image)
    edges = cv2.Canny(image, 100, 200)
    edges = np.repeat(edges[:, :, None], repeats=3, axis=2)
    canny_image = load_image(edges)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        image=canny_image,
        num_inference_steps=20,
        guidance_scale=7.5,
    ).images[0]

    output_file = Path(output_path)
    output.save(output_file)
    return str(output_file)


if __name__ == "__main__":
    # Example:
    # generate_controlled_image(
    #     subject="female android",
    #     scene="cyberpunk city at night",
    #     style="cinematic concept art",
    #     reference_image_path="assets/input.jpg",
    #     output_path="vibe_generated_output.png",
    # )
    pass
