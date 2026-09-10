from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
import torch
from src.prompt_engine import PromptComposer, GeneratedPrompt
from src import sd_diffusers_pipeline as module


def test_default_matrix_is_independent_of_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "cinematic_portrait" in PromptComposer().list_templates()


def test_midjourney_does_not_receive_sd_weight_syntax():
    prompt = PromptComposer().compose("cinematic_portrait", "robot", target_platform="midjourney", mj_param_preset="v6_photoreal")
    assert "(worst" not in prompt.as_midjourney_string()
    assert "--v 6.0" in prompt.as_midjourney_string()
    assert "--no" not in GeneratedPrompt("robot", "").as_midjourney_string()


def test_invalid_platform_and_preset_fail():
    with pytest.raises(ValueError):
        PromptComposer().compose("cinematic_portrait", "robot", target_platform="typo")
    with pytest.raises(KeyError):
        PromptComposer().compose("cinematic_portrait", "robot", target_platform="midjourney", mj_param_preset="typo")


def test_cpu_pipeline_uses_float32(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    loaders = []
    for cls in [module.AutoencoderKL, module.ControlNetModel, module.StableDiffusionControlNetPipeline]:
        loader = Mock()
        monkeypatch.setattr(cls, "from_pretrained", loader)
        loaders.append(loader)
    monkeypatch.setattr(module.UniPCMultistepScheduler, "from_config", Mock())
    pipeline = module.VibeAIGCPipeline()
    pipeline._initialize_pipeline("test-controlnet")
    assert all(loader.call_args.kwargs["torch_dtype"] == torch.float32 for loader in loaders)
    loaders[-1].return_value.to.assert_called_once_with("cpu")
    loaders[-1].return_value.enable_model_cpu_offload.assert_not_called()


def test_canny_passes_pil_image_and_seed(tmp_path):
    reference = tmp_path / "input.png"
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[8:24, 8:24] = 255
    Image.fromarray(pixels).save(reference)
    pipeline = module.VibeAIGCPipeline()
    pipeline.pipeline = Mock(return_value=SimpleNamespace(images=[Image.new("RGB", (32, 32))]))
    output = tmp_path / "nested" / "result.png"
    pipeline.run_canny_generation("cinematic_portrait", "robot", str(reference), str(output), seed=17)
    kwargs = pipeline.pipeline.call_args.kwargs
    assert isinstance(kwargs["image"], Image.Image)
    assert kwargs["image"].mode == "RGB"
    assert np.array(kwargs["image"]).max() == 255
    assert kwargs["generator"].initial_seed() == 17
    assert output.exists()
