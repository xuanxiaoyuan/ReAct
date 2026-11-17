import json
import os
import unittest
from unittest.mock import patch

from react_agent.agent import Agent, AgentType
from react_agent.client import OpenAICompatibleClient
from react_agent.environment import DEFAULT_TASK, MiniHouse


def call(call_id: str, name: str, **arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }],
    }


class ScriptedModel:
    def __init__(self, messages):
        self.responses = iter(messages)
        self.seen = []

    def complete(self, messages, tools):
        self.seen.append((list(messages), tools))
        return next(self.responses)


class AgentLoopTests(unittest.TestCase):
    def test_dashscope_environment_defaults(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}, clear=True):
            client = OpenAICompatibleClient.from_env()
        self.assertEqual(client.model, "qwen-plus")
        self.assertEqual(client.base_url, "https://dashscope.aliyuncs.com/compatible-mode/v1")

    def test_full_action_observation_loop(self):
        script = [
            call("1", "observe"),
            call("2", "go", room="kitchen"),
            call("3", "take", item="apple"),
            call("4", "put", item="apple", container="fridge"),
            call("5", "go", room="hall"),
            call("6", "go", room="study"),
            call("7", "take", item="mug"),
            call("8", "go", room="hall"),
            call("9", "go", room="kitchen"),
            call("10", "clean", item="mug"),
            call("11", "go", room="hall"),
            call("12", "go", room="study"),
            call("13", "put", item="mug", container="cabinet"),
            call("14", "finish", answer="done"),
        ]
        model = ScriptedModel(script)
        house = MiniHouse()
        result = Agent(model, house.registry(), AgentType.REACT, max_steps=20).run(DEFAULT_TASK)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "done")
        self.assertTrue(house.task_succeeded())
        second_request = model.seen[1][0]
        self.assertEqual(second_request[-1]["role"], "tool")
        self.assertEqual(second_request[-1]["tool_call_id"], "1")

    def test_premature_finish_is_an_observation(self):
        model = ScriptedModel([call("1", "finish", answer="done")])
        house = MiniHouse()
        result = Agent(model, house.registry(), max_steps=1).run(DEFAULT_TASK)
        self.assertEqual(result.status, "max_steps_exceeded")
        self.assertFalse(house.finished)
        observation = json.loads(result.messages[-1]["content"])
        self.assertFalse(observation["ok"])

    def test_each_mode_selects_its_prompt(self):
        for mode in AgentType:
            with self.subTest(mode=mode):
                model = ScriptedModel([{"role": "assistant", "content": "answer"}])
                result = Agent(model, MiniHouse().registry(), mode, max_steps=1).run("task")
                prompt = model.seen[0][0][0]["content"]
                expected = {
                    AgentType.ACT_ONLY: "Mode: Act-only",
                    AgentType.REACT: "Mode: ReAct (",
                    AgentType.REACT_IM: "Mode: ReAct-IM",
                }[mode]
                self.assertIn(expected, prompt)
                self.assertEqual(result.status, "max_steps_exceeded")

    def test_plain_text_is_not_false_success(self):
        model = ScriptedModel([{"role": "assistant", "content": "I am done"}])
        result = Agent(model, MiniHouse().registry(), max_steps=1).run(DEFAULT_TASK)
        self.assertEqual(result.status, "max_steps_exceeded")
        self.assertIn("Call finish", result.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
