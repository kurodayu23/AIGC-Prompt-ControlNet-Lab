[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md)

---

# 🎨 AIGC Prompt Engineering & ControlNet MLOps Lab

> **A production-grade bridge between unstructured artistic prompting and deterministic engineering pipelines.**

Most generative AI repositories fall into two extremes: they are either pure text files of prompts, or heavily tangled Jupiter Notebooks. This repository demonstrates the **"Generative AI Engineering Capability"** by unifying dynamic prompt compilation (Midjourney V6 & SDXL) with a programmatic Stable Diffusion ControlNet pipeline.

## 🏗️ Core Architecture & Capabilities

This lab is split into two complementary engines:

### 1. The Prompt Composition Engine (`src/prompt_engine.py`)
A deterministic prompt compiler that abstracts away the syntax differences between generative platforms.

*   **Syntax Translation**: Dynamically compiles token weights (`(token:1.5)` for SD vs `token::1.5` for Midjourney).
*   **Universal Negatives**: Centralized negative prompt injection to ensure baseline quality across all generations.
*   **Midjourney Parameter Serialization**: Automatically appends V6 aspect ratios, styling, and chaos values (`--ar 16:9 --v 6.0 --s 250`).
*   **JSON-Backed Matrix**: Decouples prompt design from codebase, allowing non-technical artists to update the `prompts/midjourney_prompt_matrix.json` without breaking the Python orchestration.

### 2. Vibe AIGC Pipeline (`src/sd_diffusers_pipeline.py`)
A fully-fledged `diffusers` pipeline demonstrating MLOps methodologies tailored for local GPU inference.

*   **VRAM Footprint Optimization**: Implements sequential CPU offloading (`enable_model_cpu_offload()`) and Memory Efficient Attention via `xformers`. Allows 8GB VRAM cards to process complex ControlNet graphs.
*   **Canny Edge Extraction**: Uses `OpenCV` (cv2) to automatically extract outlines from reference images for structural conditioning.
*   **High-Fidelity Autoencoders**: Hot-swaps the default VAE with `stabilityai/sd-vae-ft-mse` to prevent visual color shifting during deterministic seed generation.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Recommended: Create a conda/venv environment first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate xformers opencv-python
```

### 2. Compiling Prompts (Cross-Platform)

```python
from prompt_engine import PromptComposer
import json

composer = PromptComposer()

# Compile for Stable Diffusion (handles token weighting & structural negatives)
sd_prompt = composer.compose(
    template_name="cyberpunk_concept", 
    subject="female android walking", 
    extra_positives=["trending on artstation", composer.apply_weight("lens flare", 1.5, "sd")], 
    target_platform="sd"
)
print(json.dumps(sd_prompt.as_sd_payload(), indent=2))

# Compile for Midjourney (handles parameter injection)
mj_prompt = composer.compose(
    template_name="cinematic_portrait", 
    subject="an elven ranger", 
    target_platform="midjourney", 
    mj_param_preset="v6_photoreal"
)
print(mj_prompt.as_midjourney_string()) 
# Output: /imagine prompt: Cinematic portrait shot of an elven ranger... --v 6.0 --style raw --ar 16:9 --q 2 --s 250
```

### 3. Programmatic Image Generation (SD + ControlNet)

```python
from sd_diffusers_pipeline import VibeAIGCPipeline

pipeline = VibeAIGCPipeline(use_xformers=True)

output_path = pipeline.run_canny_generation(
    template="cyberpunk_concept",
    subject="female android walking",
    reference_image_path="assets/input.jpg",
    output_path="assets/showcase_out.png",
    seed=8848
)
```

## 🧠 Why This Matters (The Engineering Perspective)
In enterprise workflows, prompt generation cannot be manual, and image generation cannot rely entirely on a UI (like WebUI/ComfyUI) if you want to deploy it as a microservice.

1.  **Reproducibility**: By controlling the seed, the generation parameters, and the prompt dynamically via Python classes, we guarantee deterministic tests for the AI output.
2.  **Modularity**: The prompt matrix JSON behaves like a configuration map. You can deploy updates to prompt art direction without demanding a zero-downtime redeploy of the generative microservices.
3.  **Hardware Awareness**: The pipeline explicitly checks OS device capabilities (CUDA fallback) and injects RAM/VRAM offloading constraints.

*(This repository demonstrates advanced understanding of multimodal orchestration patterns).*
