from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from perception_layer.path_config import DEFAULT_CONFIG_PATH, read_config


def print_phase(title: str, content: object) -> None:
    print(f"\n[{title}]\n{content}\n", flush=True)


def load_vision_model(config_path: str | Path = DEFAULT_CONFIG_PATH) -> str:
    config = read_config(config_path)
    model = config.get("thread_1", "vision_model", fallback="").strip()
    if not model:
        raise ValueError("Missing [thread_1] vision_model in perception config.")
    return model


class HighlightVisionAgent:
    def __init__(
        self,
        model: str | None = None,
        timeout: int = 180,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model = model or load_vision_model(config_path)
        self.timeout = timeout
        self.prompt_path = Path(__file__).with_name("vision_prompt.md")

    def build_prompt(self, candidate_ranges: str, context: str = "") -> str:
        prompt = self.prompt_path.read_text(encoding="utf-8")
        return (
            prompt
            .replace("{{candidate_ranges}}", candidate_ranges.strip() or "No candidate ranges.")
            .replace("{{context}}", context.strip() or "No additional context.")
        )

    def run(self, image_path: str | Path, candidate_ranges: str, context: str = "") -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

        image_path = Path(image_path)
        mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        prompt = self.build_prompt(candidate_ranges, context)
        print_phase(
            "THREAD 1 VISION INPUT",
            "\n".join([
                f"model: {self.model}",
                f"image_path: {image_path}",
                f"candidate_ranges: {candidate_ranges}",
                f"context: {context or 'No additional context.'}",
                "",
                "prompt:",
                prompt,
            ]),
        )

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a precise wearable sensor visual anomaly verifier."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Image: {image_path.name}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API Error {response.status_code}: {response.text}")
        output = response.json()["choices"][0]["message"]["content"].strip()
        print_phase("THREAD 1 VISION OUTPUT", output)
        return output


class VisionOnlyAgent:
    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        model: str | None = None,
        timeout: int = 180,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model = model or load_vision_model(config_path)
        self.timeout = timeout
        self.prompt_path = Path(__file__).with_name("vision_only_prompt.md")

    def run(self, image_folder: str | Path) -> str:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

        image_folder = Path(image_folder)
        image_paths = sorted(
            path
            for path in image_folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
        )
        if not image_paths:
            raise FileNotFoundError(f"No sensor images found in {image_folder}")

        content: list[dict[str, object]] = []
        for image_path in image_paths:
            mime_type = (
                "image/jpeg"
                if image_path.suffix.lower() in {".jpg", ".jpeg"}
                else "image/png"
            )
            image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
            content.extend(
                [
                    {"type": "text", "text": f"Sensor plot: {image_path.name}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{image_b64}"
                        },
                    },
                ]
            )
        content.append(
            {
                "type": "text",
                "text": self.prompt_path.read_text(encoding="utf-8"),
            }
        )
        print_phase(
            "THREAD 1 VISION ONLY INPUT",
            "\n".join([
                f"model: {self.model}",
                f"image_folder: {image_folder}",
                "image_paths:",
                *[f"- {path}" for path in image_paths],
                "",
                "prompt:",
                self.prompt_path.read_text(encoding="utf-8"),
            ]),
        )

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise wearable sensor vision analyst. "
                            "Follow the required output format exactly."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API Error {response.status_code}: {response.text}"
            )
        output = response.json()["choices"][0]["message"]["content"].strip()
        print_phase("THREAD 1 VISION ONLY OUTPUT", output)
        return output
