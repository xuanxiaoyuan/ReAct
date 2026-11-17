"""The actual agent loop. No agent framework is used."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .client import ChatModel
from .prompts import PROMPTS
from .tools import ToolRegistry


class AgentType(str, Enum):
    ACT_ONLY = "act-only"
    REACT = "react"
    REACT_IM = "react-im"


@dataclass
class AgentResult:
    answer: str
    status: str
    steps: int
    messages: list[dict[str, Any]] = field(repr=False)


class Agent:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        agent_type: AgentType | str = AgentType.REACT,
        max_steps: int = 20,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.agent_type = AgentType(agent_type)
        self.max_steps = max_steps
        self.on_event = on_event or (lambda _kind, _payload: None)

    def run(self, task: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": PROMPTS[self.agent_type.value]},
            {"role": "user", "content": task},
        ]
        for step in range(1, self.max_steps + 1):
            assistant = self.model.complete(messages, self.tools.schemas())
            assistant = self._normalise_assistant(assistant)
            messages.append(assistant)
            self.on_event("assistant", {"step": step, "message": assistant})
            calls = assistant.get("tool_calls") or []
            if not calls:
                content = (assistant.get("content") or "").strip()
                reminder = "Do not stop with a text response. Call finish if the task is complete; otherwise call the next tool."
                if content:
                    reminder += " Your unsubmitted draft was: " + content
                messages.append({"role": "user", "content": reminder})
                continue
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                observation = self.tools.execute(name, function.get("arguments", "{}"))
                tool_message = {"role": "tool", "tool_call_id": call.get("id", ""), "content": observation}
                messages.append(tool_message)
                self.on_event("tool", {"step": step, "name": name, "observation": observation})
                if name == "finish":
                    parsed = json.loads(observation)
                    if parsed.get("ok"):
                        return AgentResult(str(parsed.get("result", "")), "completed", step, messages)
        return AgentResult("", "max_steps_exceeded", self.max_steps, messages)

    @staticmethod
    def _normalise_assistant(message: dict[str, Any]) -> dict[str, Any]:
        normalised = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            normalised["tool_calls"] = message["tool_calls"]
        return normalised
