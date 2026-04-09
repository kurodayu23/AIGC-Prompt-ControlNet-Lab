# Stable Diffusion & ControlNet Automated Flow

**Status**: Generative AI Orchestration  
**Stack**: HuggingFace Diffusers, OpenCV, ControlNet, PyTorch

## Overview
A programmatic pipeline demonstrating deep integration with Stable Diffusion beyond mere UI prompting. This repository exposes how to orchestrate Diffusers, inject ControlNet boundaries (Canny edge detection), and generate consistent, structurally guided imagery entirely via code.

## Why Code > UI?
By wrapping image generation in Python functions, this enables:
1. **Batch processing** and parameter grid searches.
2. **API integration** into larger AI agent workflows.
3. **Deterministic testing** using fixed seeds and latents.
