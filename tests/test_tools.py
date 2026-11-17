import json
import unittest

from react_agent.tools import ToolRegistry, function_tool


class FunctionToolTests(unittest.TestCase):
    def test_schema_and_dispatch(self):
        @function_tool
        def add(left: int, right: int = 1) -> int:
            """Add two integers.
            left: First integer.
            right: Second integer.
            """
            return left + right

        schema = add.openai_schema()["function"]
        self.assertEqual(schema["parameters"]["properties"]["left"]["type"], "integer")
        self.assertEqual(schema["parameters"]["required"], ["left"])
        result = json.loads(ToolRegistry([add]).execute("add", '{"left": 2, "right": 3}'))
        self.assertEqual(result, {"ok": True, "result": 5})

    def test_invalid_arguments_become_observation(self):
        @function_tool
        def echo(value: str) -> str:
            """Echo text."""
            return value

        result = json.loads(ToolRegistry([echo]).execute("echo", '{"wrong": 1}'))
        self.assertFalse(result["ok"])
        self.assertIn("unknown arguments", result["error"])

    def test_wrong_type_becomes_observation(self):
        @function_tool
        def square(value: int) -> int:
            """Square an integer."""
            return value * value

        result = json.loads(ToolRegistry([square]).execute("square", '{"value": "2"}'))
        self.assertFalse(result["ok"])
        self.assertIn("must be integer", result["error"])


if __name__ == "__main__":
    unittest.main()
