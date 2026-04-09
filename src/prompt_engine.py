from __future__ import annotations

import json
from pathlib import Path


class PromptComposer:
    """Compose reproducible prompts from a template matrix."""

    def __init__(self, matrix_path: str = "prompts/midjourney_prompt_matrix.json") -> None:
        self.matrix_path = Path(matrix_path)
        self._db = json.loads(self.matrix_path.read_text(encoding="utf-8"))

    def list_templates(self) -> list[str]:
        return sorted(self._db.get("templates", {}).keys())

    def compose(self, template: str, subject: str, extra: str = "") -> str:
        templates: dict[str, str] = self._db.get("templates", {})
        if template not in templates:
            raise KeyError(f"Unknown template: {template}")
        base = templates[template].format(subject=subject)
        return f"{base}, {extra}".strip(", ")


def build_controlnet_prompt(subject: str, scene: str, style: str) -> str:
    # Stable and explicit instructions usually produce more predictable generations.
    return (
        f"{subject}, {scene}, {style}, ultra detailed, high coherence, "
        "clean composition, physically plausible lighting"
    )
