from __future__ import annotations

import argparse
import json
from typing import Any

from .agent import Agent, AgentType
from .client import OpenAICompatibleClient
from .environment import DEFAULT_TASK, MiniHouse


def _print_event(kind: str, payload: dict[str, Any]) -> None:
    if kind == "assistant":
        message = payload["message"]
        if message.get("content"):
            print(f"Thought: {message['content']}")
        for call in message.get("tool_calls", []):
            function = call["function"]
            print(f"Action: {function['name']}({function.get('arguments', '{}')})")
    else:
        observation = json.loads(payload["observation"])
        print(f"Observation: {json.dumps(observation, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Act-only, ReAct, or ReAct-IM from scratch")
    parser.add_argument("--mode", choices=[x.value for x in AgentType], default="react")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--max-steps", type=int, default=20)
    args = parser.parse_args()
    environment = MiniHouse()
    agent = Agent(
        OpenAICompatibleClient.from_env(),
        environment.registry(),
        agent_type=args.mode,
        max_steps=args.max_steps,
        on_event=_print_event,
    )
    result = agent.run(args.task)
    print(f"\nstatus={result.status}; environment_success={environment.task_succeeded()}")
    print(f"answer={result.answer}")


if __name__ == "__main__":
    main()
