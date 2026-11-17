"""A tiny OpenAI-compatible chat client using only the standard library."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]: ...


@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 90.0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        api_key = dashscope_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY or OPENAI_API_KEY is not set")
        if dashscope_key:
            default_model = "qwen-plus"
            default_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        else:
            default_model = "gpt-4.1-mini"
            default_base_url = "https://api.openai.com/v1"
        return cls(
            model=os.getenv("OPENAI_MODEL", default_model),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", default_base_url).rstrip("/"),
        )

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
        payload = json.dumps(
            {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto"},
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"model API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"model API request failed: {exc.reason}") from exc
        try:
            return body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected model response: {body}") from exc
