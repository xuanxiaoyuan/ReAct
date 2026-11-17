"""Small, framework-free agent loops."""

from .agent import Agent, AgentResult, AgentType
from .client import OpenAICompatibleClient
from .tools import Tool, ToolRegistry, function_tool

__all__ = [
    "Agent",
    "AgentResult",
    "AgentType",
    "OpenAICompatibleClient",
    "Tool",
    "ToolRegistry",
    "function_tool",
]
