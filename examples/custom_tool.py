"""Minimal example: add your own function tool to the loop."""

from react_agent import Agent, OpenAICompatibleClient, ToolRegistry, function_tool


@function_tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression containing numbers and + - * /.
    expression: Arithmetic expression, for example (12 + 3) / 5.
    """
    # A real project should use an AST parser; eval is deliberately constrained here.
    if any(character not in "0123456789.+-*/() " for character in expression):
        raise ValueError("unsupported character")
    return float(eval(expression, {"__builtins__": {}}, {}))


@function_tool
def finish(answer: str) -> str:
    """Return the final answer.
    answer: Final answer for the user.
    """
    return answer


agent = Agent(
    model=OpenAICompatibleClient.from_env(),
    tools=ToolRegistry([calculator, finish]),
    agent_type="react",
)
print(agent.run("What is (37 * 19) + 8?").answer)
