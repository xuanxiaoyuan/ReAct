"""Function tools: Python functions, JSON schemas, validation and dispatch."""

from __future__ import annotations

import inspect
import json
import types
from dataclasses import dataclass
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints


class ToolError(Exception):
    """An expected tool validation or execution error."""


def _validate_value(name: str, value: Any, schema: dict[str, Any]) -> None:
    if "anyOf" in schema:
        for candidate in schema["anyOf"]:
            try:
                _validate_value(name, value, candidate)
                return
            except ToolError:
                pass
        raise ToolError(f"argument {name} does not match any allowed type")
    expected = schema.get("type")
    matches = {
        "null": value is None,
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if expected and not matches.get(expected, True):
        raise ToolError(f"argument {name} must be {expected}, got {type(value).__name__}")
    if expected == "array" and "items" in schema:
        for index, item in enumerate(value):
            _validate_value(f"{name}[{index}]", item, schema["items"])


def _json_type(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        non_null = [item for item in args if item is not type(None)]
        schema = _json_type(non_null[0]) if len(non_null) == 1 else {"anyOf": [_json_type(x) for x in non_null]}
        if len(non_null) != len(args):
            schema = {"anyOf": [schema, {"type": "null"}]}
        return schema
    if origin in (list, tuple, set):
        return {"type": "array", "items": _json_type(args[0] if args else Any)}
    if origin is dict:
        return {"type": "object"}
    return {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }.get(annotation, {"type": "string"})


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def invoke(self, arguments: str | dict[str, Any]) -> Any:
        try:
            values = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError as exc:
            raise ToolError(f"arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(values, dict):
            raise ToolError("arguments must be a JSON object")
        unknown = set(values) - set(self.parameters.get("properties", {}))
        missing = set(self.parameters.get("required", [])) - set(values)
        if unknown:
            raise ToolError(f"unknown arguments: {', '.join(sorted(unknown))}")
        if missing:
            raise ToolError(f"missing arguments: {', '.join(sorted(missing))}")
        for argument_name, value in values.items():
            _validate_value(argument_name, value, self.parameters["properties"][argument_name])
        try:
            return self.function(**values)
        except ToolError:
            raise
        except Exception as exc:  # Tools report failures as observations; the loop stays alive.
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc


def function_tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a typed Python function into a provider-compatible function tool.

    The first line of the docstring becomes the description. Parameters may be
    documented on separate lines as ``param: explanation``.
    """

    def decorate(target: Callable[..., Any]) -> Tool:
        signature = inspect.signature(target)
        hints = get_type_hints(target)
        doc = inspect.getdoc(target) or ""
        lines = [line.strip() for line in doc.splitlines() if line.strip()]
        parameter_docs: dict[str, str] = {}
        for line in lines[1:]:
            key, separator, value = line.partition(":")
            if separator and key.strip() in signature.parameters:
                parameter_docs[key.strip()] = value.strip()
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter_name, parameter in signature.parameters.items():
            schema = _json_type(hints.get(parameter_name, parameter.annotation))
            if parameter_name in parameter_docs:
                schema["description"] = parameter_docs[parameter_name]
            properties[parameter_name] = schema
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
        return Tool(
            name=name or target.__name__,
            description=description or (lines[0] if lines else target.__name__),
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            function=target,
        )

    return decorate(func) if func is not None else decorate


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: str | dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return json.dumps({"ok": False, "error": f"unknown tool: {name}"}, ensure_ascii=False)
        try:
            value = tool.invoke(arguments)
            return json.dumps({"ok": True, "result": value}, ensure_ascii=False, default=str)
        except ToolError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
