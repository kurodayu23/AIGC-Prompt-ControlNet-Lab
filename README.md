[English](README.md) | [简体中文](README_zh.md) | [日本語](README_ja.md)

---

# AIGC Prompt + ControlNet Workflow

This repository combines two interview-relevant capabilities:
- Prompt engineering with reusable template matrix
- Programmatic Stable Diffusion + ControlNet generation

## Files

- `src/prompt_engine.py`: template-based prompt composition
- `src/sd_diffusers_pipeline.py`: ControlNet canny pipeline
- `prompts/midjourney_prompt_matrix.json`: reusable style/parameter matrix

## Notes

- Intended as a reproducible engineering demo, not a no-code GUI package.
- Requires local GPU/VRAM for practical speed.

## License

MIT
