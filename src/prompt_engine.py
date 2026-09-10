from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GeneratedPrompt:
    positive: str
    negative: str
    midjourney_suffix: str = ""
    
    def as_sd_payload(self) -> dict[str, str]:
        """Format for standard SDXL / A1111 rest API payloads."""
        return {"prompt": self.positive, "negative_prompt": self.negative}
        
    def as_midjourney_string(self) -> str:
        """Format for Discord/Midjourney Bot invocation."""
        negative = f" --no {self.negative}" if self.negative else ""
        return f"/imagine prompt: {self.positive}{negative} {self.midjourney_suffix}".strip()


class PromptComposer:
    """从 JSON 模板组合 Stable Diffusion 和 Midjourney 提示词。"""

    def __init__(self, matrix_path: str | None = None) -> None:
        self.matrix_path = (Path(matrix_path) if matrix_path is not None else
                            Path(__file__).resolve().parents[1] / "prompts/midjourney_prompt_matrix.json")
        self._db = json.loads(self.matrix_path.read_text(encoding="utf-8"))

    def list_templates(self) -> list[str]:
        return sorted(self._db.get("templates", {}).keys())

    def apply_weight(self, token: str, weight: float, engine: str = "sd") -> str:
        """
        Dynamically applies attention weights based on target generative engine.
        SD uses (token:1.5), Midjourney uses ::1.5 
        """
        if engine.lower() == "sd":
            return f"({token}:{weight:.1f})"
        elif engine.lower() == "midjourney":
            return f"{token}::{weight:.1f}"
        raise ValueError(f"Unknown engine: {engine}")

    def compose(
        self, 
        template_name: str, 
        subject: str, 
        extra_positives: list[str] | None = None,
        target_platform: str = "sd",
        mj_param_preset: str | None = None
    ) -> GeneratedPrompt:
        """
        Constructs a deterministic programmatic prompt tailored to the target model.
        """
        if target_platform.lower() not in {"sd", "midjourney"}:
            raise ValueError(f"Unknown platform: {target_platform}")
        templates: dict[str, str] = self._db.get("templates", {})
        if template_name not in templates:
            raise KeyError(f"Unknown template: {template_name}")
            
        base = templates[template_name].format(subject=subject)
        
        if extra_positives:
            base = f"{base}, {', '.join(extra_positives)}"
            
        negative_prompt = self._db.get("sd_defaults", {}).get("negative_prompt", "")
        
        if target_platform.lower() == "midjourney":
            negative_prompt = self._db.get("midjourney_negative_prompt", "")
        mj_suffix = ""
        if target_platform.lower() == "midjourney" and mj_param_preset:
            presets = self._db.get("midjourney_params", {})
            mj_suffix = presets[mj_param_preset]

        return GeneratedPrompt(
            positive=base.strip(", "),
            negative=negative_prompt,
            midjourney_suffix=mj_suffix
        )


def build_controlnet_prompt(subject: str, scene: str, style: str) -> str:
    """Legacy helper for simple ControlNet validations. Uses explicit syntax."""
    return (
        f"{subject}, {scene}, {style}, ultra detailed, (high coherence:1.2), "
        "clean composition, physically plausible lighting, 8k resolution"
    )

if __name__ == "__main__":
    composer = PromptComposer()
    print("--- SDXL Payload ---")
    sd_prompt = composer.compose("cyberpunk_concept", "a futuristic sports car", ["trending on artstation", composer.apply_weight("lens flare", 1.5, "sd")], target_platform="sd")
    print(json.dumps(sd_prompt.as_sd_payload(), indent=2))
    
    print("\n--- Midjourney V6 Payload ---")
    mj_prompt = composer.compose("cinematic_portrait", "an elven ranger", target_platform="midjourney", mj_param_preset="v6_photoreal")
    print(mj_prompt.as_midjourney_string())
