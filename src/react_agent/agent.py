"""The actual agent loop. No agent framework is used."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from .client import ChatModel
from .prompts import initial_messages
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
    trace: list[dict[str, Any]] = field(default_factory=list)


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
        messages = initial_messages(self.agent_type.value, task)
        trace: list[dict[str, Any]] = []
        for step in range(1, self.max_steps + 1):
            if self.agent_type is AgentType.REACT_IM:
                thought_reply = self.model.complete(
                    [*messages, {"role": "user", "content": "Inner monologue for the next action: state goal and immediate subgoal only."}],
                    [],
                )
                thought = (thought_reply.get("content") or "").strip()
                if not thought or thought_reply.get("tool_calls"):
                    messages.append({"role": "user", "content": "Provide the required inner monologue before acting."})
                    continue
                messages.append({"role": "assistant", "content": thought})
                trace.append({"step": step, "kind": "thought", "text": thought})
                self.on_event("thought", {"step": step, "text": thought})
            assistant = self.model.complete(messages, self.tools.schemas())
            assistant = self._normalise_assistant(assistant)
            messages.append(assistant)
            self.on_event("assistant", {"step": step, "message": assistant})
            content = (assistant.get("content") or "").strip()
            if self.agent_type is AgentType.REACT and content:
                trace.append({"step": step, "kind": "thought", "text": content})
                self.on_event("thought", {"step": step, "text": content})
            calls = assistant.get("tool_calls") or []
            if not calls:
                reminder = "Do not stop with a text response. Call finish if the task is complete; otherwise call the next tool."
                if content:
                    reminder += " Your unsubmitted draft was: " + content
                messages.append({"role": "user", "content": reminder})
                continue
            if len(calls) != 1:
                # One action per cycle keeps observation ordering unambiguous.
                for call in calls:
                    messages.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                     "content": json.dumps({"ok": False, "error": "call exactly one tool per step"})})
                continue
            for call in calls:
                function = call.get("function", {})
                name = function.get("name", "")
                trace.append({"step": step, "kind": "action", "name": name,
                              "arguments": function.get("arguments", "{}")})
                observation = self.tools.execute(name, function.get("arguments", "{}"))
                tool_message = {"role": "tool", "tool_call_id": call.get("id", ""), "content": observation}
                messages.append(tool_message)
                trace.append({"step": step, "kind": "observation", "name": name, "value": observation})
                self.on_event("tool", {"step": step, "name": name, "observation": observation})
                if name == "finish":
                    parsed = json.loads(observation)
                    if parsed.get("ok"):
                        return AgentResult(str(parsed.get("result", "")), "completed", step, messages, trace)
        return AgentResult("", "max_steps_exceeded", self.max_steps, messages, trace)

    @staticmethod
    def _normalise_assistant(message: dict[str, Any]) -> dict[str, Any]:
        normalised = {"role": "assistant", "content": message.get("content")}
        if message.get("tool_calls"):
            normalised["tool_calls"] = message["tool_calls"]
        return normalised
