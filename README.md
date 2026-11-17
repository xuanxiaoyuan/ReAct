# 从零实现 Act-only、ReAct 与 ReAct-IM

这是一个不依赖 LangChain / Agent SDK 的教学实现。重点是把 agent loop 中真正发生的事展示出来：

```text
用户任务
   ↓
LLM(messages, tool schemas)
   ├─ 普通文本 ───────────────→ 最终回答
   └─ function tool call
          ↓
      校验参数 → 执行 Python 函数
          ↓
      Observation(role=tool)
          └───────────────────→ 下一轮 LLM
```

项目实现了三种论文中的策略：

| 模式 | 模型在两次动作之间生成什么 | 用途 |
|---|---|---|
| `act-only` | 不输出 reasoning，直接选择工具 | 论文中的无推理基线 |
| `react` | 稀疏且灵活的 Thought，可规划、更新和处理异常 | 标准 ReAct |
| `react-im` | 每步都输出密集反馈，只描述目标拆解和当前子目标 | 论文的 Inner Monologue 消融 |

注意：ReAct-IM 里的 IM 是 **Inner Monologue**，不是 memory。这个区别来自 ReAct 论文在
ALFWorld 上的消融设置。

## 目录

```text
src/react_agent/
  agent.py        # 共享的 agent loop
  tools.py        # @function_tool、JSON Schema、参数校验、工具分发
  client.py       # 零依赖 OpenAI-compatible Chat Completions 客户端
  prompts.py      # 三种策略唯一有意不同的部分
  environment.py  # 可复现的 ALFWorld 风格本地环境
  cli.py          # 命令行入口与轨迹输出
tests/            # 不调用外部 API 的 loop / tools 测试
examples/         # 自定义 function tool 示例
```

## 运行

Python 3.10+ 即可，项目本身没有第三方运行时依赖。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

PowerShell 不会自动读取 `.env`。阿里云百炼按量付费 Key（`sk-` 开头）可这样配置：

```powershell
$env:DASHSCOPE_API_KEY = "你的百炼 API Key"
$env:OPENAI_MODEL = "qwen-plus"
```

检测到 `DASHSCOPE_API_KEY` 后，客户端会自动使用百炼按量付费地址
`https://dashscope.aliyuncs.com/compatible-mode/v1`。Token Plan、Coding Plan 的 Key 和地址不同，
不能与这个地址混用。任何其他兼容 `/chat/completions` 且支持 native tool calling 的服务也可以
通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 和 `OPENAI_MODEL` 接入。

```powershell
react-agent --mode act-only
react-agent --mode react
react-agent --mode react-im
```

默认任务是：把苹果放进厨房冰箱，然后清洗杯子并放进书房柜子。环境包含 `observe`、
`go`、`take`、`clean`、`put`、`finish` 六个 function tools。最后的 `finish` 也会检查真实
环境状态，普通文本不会被当成成功，agent 不能靠一句 “done” 假装完成。

不安装项目也可直接运行：

```powershell
$env:PYTHONPATH = "src"
python -m react_agent.cli --mode react --max-steps 20
```

## 自己写 function tool

```python
from react_agent import ToolRegistry, function_tool

@function_tool
def search(query: str, limit: int = 5) -> list[str]:
    """Search the local knowledge base.
    query: Search terms.
    limit: Maximum number of results.
    """
    return my_index.search(query, limit=limit)

tools = ToolRegistry([search])
```

装饰器利用类型注解和 docstring 生成 provider 接受的 JSON Schema。`ToolRegistry` 负责拒绝
缺失/多余参数、捕获预期错误，并把结果统一编码成 Observation：

```json
{"ok": true, "result": ["..."]}
```

完整的计算器例子见 `examples/custom_tool.py`。

## 测试

测试使用 ScriptedModel 模拟模型逐步发出 tool calls，因此不需要 API key：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## 与 2023 原始实现的差异

原论文用文本格式 `Thought / Action / Observation` 再手工解析动作；这里保留同样的策略差异，
但动作改用现在常见的 native function calling。这样参数有 JSON Schema 约束，tool call ID 和
Observation 能可靠配对，也更接近今天实际开发 agent loop 的方式。

论文：Yao et al., [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
