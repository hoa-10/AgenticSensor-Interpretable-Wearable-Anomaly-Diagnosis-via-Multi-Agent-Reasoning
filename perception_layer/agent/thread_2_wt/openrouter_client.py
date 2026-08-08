from __future__ import annotations

import os
from typing import Any

import requests


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


class OpenRouterClient:
    """Small OpenRouter chat client with the run_turn contract used by Thread 2."""

    def __init__(
        self,
        system_prompt: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 180,
    ) -> None:
        self.system_prompt = system_prompt
        self.model = model
        self.timeout = timeout

    def run_turn(self, prompt: str) -> dict[str, Any]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
            },
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter API Error {response.status_code}: {response.text}"
            )

        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return {"text": str(content or "").strip(), "response": payload}
