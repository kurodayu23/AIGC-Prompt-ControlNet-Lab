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

from PIL import Image

if __package__:
    from .prompt_engine import PromptComposer
else:
    from prompt_engine import PromptComposer


logger = logging.getLogger(__name__)


class VibeAIGCPipeline:
    """Stable Diffusion 1.5 的 Canny ControlNet 调用示例。"""

    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5", use_xformers: bool = False):
        self.model_id = model_id
        self.use_xformers = use_xformers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline: Optional[StableDiffusionControlNetPipeline] = None
        self.composer = PromptComposer()
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    def _initialize_pipeline(self, controlnet_id: str) -> None:
        """Lazy load pipeline with memory optimizations."""
        logger.info(f"Initializing VAE and ControlNet: {controlnet_id}")
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=self.dtype)
        
        controlnet = ControlNetModel.from_pretrained(
            controlnet_id,
            torch_dtype=self.dtype,
        )
        
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=self.dtype,
        )
        
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        
        # VRAM Optimization Strategies
        if self.device.type == "cuda":
            logger.info("Applying GPU memory optimizations (Model CPU Offload)")
            self.pipeline.enable_model_cpu_offload()
            if self.use_xformers:
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("xformers attention enabled.")
                except ImportError:
                    logger.warning("xformers not installed. Falling back to standard attention.")
        else:
            self.pipeline.to("cpu")

    def run_canny_generation(
        self, 
        template: str, 
        subject: str, 
        reference_image_path: str, 
        output_path: str = "output/vibe_canny_out.png",
        seed: int = 42
    ) -> str:
        """
        根据参考图边缘和提示词模板生成图片。
        """
        # 1. Image Pre-processing
        logger.info(f"Extracting Canny contours from {reference_image_path}")
        image = load_image(reference_image_path)
        image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(image, 100, 200)
        edges = np.repeat(edges[:, :, None], repeats=3, axis=2)
        canny_image = Image.fromarray(edges)

        # 2. Prompt Compilation
        compiled_prompt = self.composer.compose(template_name=template, subject=subject, target_platform="sd")
        logger.info(f"Prompt Compiled:\n[+] {compiled_prompt.positive}\n[-] {compiled_prompt.negative}")

        if self.pipeline is None:
            self._initialize_pipeline("lllyasviel/sd-controlnet-canny")

        # 固定 seed 便于同环境复现，不保证跨硬件逐像素一致。
        generator = torch.Generator(device="cpu").manual_seed(seed)
        
        logger.info("Starting inference loop...")
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
        logger.info(f"Generation successful. Output saved to {output_file.absolute()}")
        
        return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Canny ControlNet 图片生成示例")
    parser.add_argument("reference_image")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--template", default="cyberpunk_concept")
    parser.add_argument("--output", default="output/canny.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    VibeAIGCPipeline().run_canny_generation(
        args.template, args.subject, args.reference_image, args.output, args.seed
    )
