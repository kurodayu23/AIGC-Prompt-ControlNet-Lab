from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel, 
    StableDiffusionControlNetPipeline, 
    UniPCMultistepScheduler,
    AutoencoderKL
)
from diffusers.utils import load_image

from prompt_engine import PromptComposer


logging.basicConfig(level=logging.INFO, format="[Vibe-AIGC-Pipeline] %(message)s")


class VibeAIGCPipeline:
    """
    Production-grade image generation pipeline demonstrating MLOps methodologies.
    Handles VRAM footprint optimization, dynamic ControlNet injection, and deterministic seeds.
    """
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", use_xformers: bool = True):
        self.model_id = model_id
        self.use_xformers = use_xformers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline: Optional[StableDiffusionControlNetPipeline] = None
        self.composer = PromptComposer()

    def _initialize_pipeline(self, controlnet_id: str) -> None:
        """Lazy load pipeline with memory optimizations."""
        logging.info(f"Initializing VAE and ControlNet: {controlnet_id}")
        
        # Load high quality VAE to prevent color shifting
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=torch.float16,
        )
        
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            safety_checker=None  # Disable to save RAM overhead during controlled generation
        )
        
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        
        # VRAM Optimization Strategies
        if self.device.type == "cuda":
            logging.info("Applying GPU memory optimizations (Sequential CPU Offload)")
            self.pipeline.enable_model_cpu_offload()
            if self.use_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logging.info("xformers attention enabled.")
                except ImportError:
                    logging.warning("xformers not installed. Falling back to standard attention.")

    def run_canny_generation(
        self, 
        template: str, 
        subject: str, 
        reference_image_path: str, 
        output_path: str = "output/vibe_canny_out.png",
        seed: int = 42
    ) -> str:
        """
        Executes a deterministic Canny-guided generation based on prompt matrix templates.
        """
        if self.pipeline is None:
            self._initialize_pipeline("lllyasviel/sd-controlnet-canny")

        # 1. Image Pre-processing
        logging.info(f"Extracting Canny contours from {reference_image_path}")
        image = load_image(reference_image_path)
        image = np.array(image)
        edges = cv2.Canny(image, 100, 200)
        edges = np.repeat(edges[:, :, None], repeats=3, axis=2)
        canny_image = load_image(edges)

        # 2. Prompt Compilation
        compiled_prompt = self.composer.compose(template_name=template, subject=subject, target_platform="sd")
        logging.info(f"Prompt Compiled:\n[+] {compiled_prompt.positive}\n[-] {compiled_prompt.negative}")

        # 3. Deterministic Generation
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logging.info("Starting inference loop...")
        output = self.pipeline(
            prompt=compiled_prompt.positive,
            negative_prompt=compiled_prompt.negative,
            image=canny_image,
            num_inference_steps=25,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        # 4. Save and return
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output.save(output_file)
        logging.info(f"Generation successful. Output saved to {output_file.absolute()}")
        
        return str(output_file)


if __name__ == "__main__":
    # Demonstrated Usage
    # pipeline = VibeAIGCPipeline()
    # pipeline.run_canny_generation(
    #     template="cyberpunk_concept",
    #     subject="female android walking",
    #     reference_image_path="assets/input.jpg",
    #     output_path="assets/showcase_v1.png",
    #     seed=8848
    # )
    pass
