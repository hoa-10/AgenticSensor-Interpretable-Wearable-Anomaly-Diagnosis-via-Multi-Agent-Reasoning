from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import requests

from perception_layer.path_config import DEFAULT_CONFIG_PATH
from perception_layer.timing import record_timing

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def collect_image_paths(image_folder_path: str | Path) -> list[Path]:
    image_folder = Path(image_folder_path)
    if not image_folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_folder}")
    if not image_folder.is_dir():
        raise ValueError(f"Expected an image folder, got: {image_folder}")

    image_paths = sorted(
        path
        for path in image_folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {image_folder}")
    return image_paths


class ImageFolderVisionAgent:
    def __init__(
        self,
        model: str | None = None,
        timeout: int = 180,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.prompt_path = Path(__file__).with_name("prompt.md")
        # Hardcoding the requested gemini model directly here
        self.model = model or "google/gemini-3-flash-preview"
        self.timeout = timeout
        self.config_path = config_path

    def build_prompt(
        self,
        context: str = "",
        image_paths: list[Path] | None = None,
        anomaly_ranges: str = "",
    ) -> str:
        prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        image_list = "\n".join(
            f"Image {index}: {image_path.name}"
            for index, image_path in enumerate(image_paths or [], start=1)
        )
        replacements = {
            "{{context}}": context.strip() or "No additional context provided.",
            "{{image_list}}": image_list or "No image list provided.",
            "{{anomaly_ranges}}": anomaly_ranges.strip() or "No specific anomaly ranges provided.",
        }
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        return prompt

    def run(
        self,
        image_folder_path: str | Path,
        context: str = "",
        anomaly_ranges: str = "",
    ) -> str:
        started_at = time.perf_counter()
        image_paths = collect_image_paths(image_folder_path)
        
        # 1. Fetch the OpenRouter API Key directly
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

        prompt_text = self.build_prompt(context, image_paths, anomaly_ranges)
        
        # 2. Build the payload manually
        content = []
        for index, image_path in enumerate(image_paths, start=1):
            mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
                
            content.append({"type": "text", "text": f"Image {index} - {image_path.name}"})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img_b64}"
                }
            })
            
        # Add the text prompt to the end of the array
        content.append({"type": "text", "text": prompt_text})
        
        messages = [
            {"role": "system", "content": "You are a highly precise wearable sensor visual analyst."},
            {"role": "user", "content": content}
        ]

        # 3. Call OpenRouter directly via requests
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": self.model,
                "messages": messages,
            },
            timeout=self.timeout,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API Error {response.status_code}: {response.text}")
            
        data = response.json()
        result = data["choices"][0]["message"]["content"]

        record_timing(
            "thread_1.image_folder_vision_agent",
            time.perf_counter() - started_at,
            image_count=len(image_paths),
            model=self.model,
        )
        return result
