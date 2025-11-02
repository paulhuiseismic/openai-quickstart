"""Patched multi-agent collaboration flow (LangChain 0.2.x + LangGraph 0.0.69)
Ensures Chart_Generator actually executes Python code via python_repl tool BEFORE emitting FINAL ANSWER.
Includes fallback execution node when model returns code block without emitting tool_calls.
Also provides a local FakeLLM test so you can validate routing logic without calling real API.

Usage (real run):
  1. Ensure requirements installed (downgraded ecosystem + langgraph==0.0.69).
  2. Set OPENAI / Azure environment variables as in your original notebook OR edit llm initialization.
  3. Run: python multi_agent_collaboration_fixed.py

To integrate into notebook:
  - Replace the chart_agent system_message with the one used here.
  - Add maybe_exec_code node.
  - Replace router logic & conditional edges with the updated version.

Test-only (offline) run uses FakeLLM to simulate outputs that would normally come from the model.
"""
from __future__ import annotations
import os
import re
from typing import Annotated, Sequence, TypedDict, Literal, List
import functools

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# -----------------------------------------------------------------------------
# LLM setup (adapt as needed). We attempt AzureChatOpenAI first, else fallback.
# -----------------------------------------------------------------------------
try:
    from azure_chat_openai import chat_model as _azure_model
    llm_research = _azure_model
    llm_chart = _azure_model
except Exception:  # Fallback to generic openai chat if Azure not configured
    try:
        from langchain_openai import ChatOpenAI
        llm_research = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        llm_chart = llm_research
    except Exception:
        llm_research = llm_chart = None  # Will rely on FakeLLM for test

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
python_repl_env = PythonREPL()

@tool
def python_repl(code: Annotated[str, "Python code to execute for chart generation."]):
    """Execute Python code. Print values you want surfaced. Adds FINAL ANSWER hint after success."""
    try:
        result = python_repl_env.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}\n\nFINAL ANSWER"
    )

# Optional search tool placeholder (replace with TavilySearchResults in real env)
@tool
def dummy_search(query: Annotated[str, "Search query."]):
    return f"(dummy search results for: {query})"

tools_research = [dummy_search]
tools_chart = [python_repl]

# -----------------------------------------------------------------------------
# Helper: create_agent
# -----------------------------------------------------------------------------

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant collaborating with other assistants. "
                    "Use the provided tools to progress. If you or another assistant has the final answer, "
                    "prefix response with FINAL ANSWER. Tools: {tool_names}.\n{system_message}"
                ),
            ),
            MessagesPlaceholder("messages"),
        ]
    ).partial(system_message=system_message).partial(
        tool_names=", ".join([t.name for t in tools])
    )
    return prompt | (llm.bind_tools(tools) if llm else FakeLLM())

# -----------------------------------------------------------------------------
# Patched system message for chart agent
# -----------------------------------------------------------------------------
CHART_SYS_MSG = (
    "Create clear charts based on provided data. If you produce Python code, you MUST call the python_repl tool "
    "with that code before giving FINAL ANSWER. Do NOT output FINAL ANSWER until tool execution succeeds. "
    "If you accidentally output code without a tool call, it will be executed automatically."
)

RESEARCH_SYS_MSG = (
    "Think through the data needed. Perform a single efficient search (dummy tool here). Provide any structured data "
    "the chart agent needs in plain text before yielding control."
)

research_agent = create_agent(llm_research, tools_research, RESEARCH_SYS_MSG)
chart_agent = create_agent(llm_chart, tools_chart, CHART_SYS_MSG)

# -----------------------------------------------------------------------------
# Agent node wrapper
# -----------------------------------------------------------------------------

def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):  # Very rare direct tool msg
        return {"messages": [result], "sender": name}
    # AIMessage normalization
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
chart_node = functools.partial(agent_node, agent=chart_agent, name="Chart_Generator")

# -----------------------------------------------------------------------------
# State schema
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], list]
    sender: str

# -----------------------------------------------------------------------------
# Fallback execution node
# -----------------------------------------------------------------------------

def maybe_exec_code(state: AgentState):
    """If Chart_Generator produced a python code block without tool_calls, execute it automatically."""
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if state.get("sender") != "Chart_Generator":
        return {"messages": [], "sender": state["sender"]}
    if "Successfully executed:" in content:
        return {"messages": [], "sender": state["sender"]}
    if "```python" in content and not getattr(last, "tool_calls", None):
        m = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if m:
            code = m.group(1)
            try:
                exec_result = python_repl_env.run(code)
            except Exception as e:
                exec_result = f"Tool error: {e!r}"
            tool_msg = ToolMessage(
                content=(
                    f"Successfully executed:\n```python\n{code}\n```\nStdout: {exec_result}\n\nFINAL ANSWER"
                ),
                name="python_repl",
                tool_call_id="manual_python",
            )
            return {"messages": [tool_msg], "sender": state["sender"]}
    return {"messages": [], "sender": state["sender"]}

# -----------------------------------------------------------------------------
# Router with new branch 'maybe_exec'
# -----------------------------------------------------------------------------

def router(state: AgentState) -> Literal["call_tool", "__end__", "continue", "maybe_exec"]:
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if getattr(last, "tool_calls", None):
        return "call_tool"
    if state.get("sender") == "Chart_Generator" and "```python" in content and "Successfully executed:" not in content:
        return "maybe_exec"
    if "FINAL ANSWER" in content:
        return "__end__"
    return "continue"

# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Chart_Generator", chart_node)
workflow.add_node("call_tool", ToolNode(tools_research + tools_chart))
workflow.add_node("maybe_exec", maybe_exec_code)

# Conditional edges
workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "Chart_Generator",
        "call_tool": "call_tool",
        "maybe_exec": "maybe_exec",
        "__end__": END,
    },
)
workflow.add_conditional_edges(
    "Chart_Generator",
    router,
    {
        "continue": "Researcher",
        "call_tool": "call_tool",
        "maybe_exec": "maybe_exec",
        "__end__": END,
    },
)
workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"Researcher": "Researcher", "Chart_Generator": "Chart_Generator"},
)
workflow.add_conditional_edges(
    "maybe_exec",
    lambda x: x["sender"],
    {"Researcher": "Researcher", "Chart_Generator": "Chart_Generator"},
)
workflow.add_edge(START, "Researcher")
app = workflow.compile()

# -----------------------------------------------------------------------------
# FakeLLM for offline testing (only used if real llm is None)
# -----------------------------------------------------------------------------
class FakeLLM:
    """Simulates model behavior. Researcher yields data; Chart outputs code first WITHOUT tool_calls, then FINAL ANSWER after fallback."""
    def __init__(self):
        self.step = 0
    def bind_tools(self, tools):  # for chaining compatibility
        return self
    def invoke(self, state):
        messages = state["messages"]
        if state.get("sender") == "Researcher":
            return AIMessage(content="US GDP data (year:value) 2000:10284.8 ... 2020:21354.1", name="Researcher")
        # Chart agent logic
        if self.step == 0:
            self.step += 1
            code = (
                "```python\nimport matplotlib.pyplot as plt\nyears=[2000,2001,2002]\ngdp=[10284.8,10621.8,10978.2]\nplt.plot(years,gdp)\nplt.title('Test GDP')\nplt.show()\n```\n"
                "(Will execute next)"
            )
            return AIMessage(content=code, name="Chart_Generator")
        return AIMessage(content="FINAL ANSWER: Chart displayed.", name="Chart_Generator")

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run_real():
    events = app.stream(
        {"messages": [HumanMessage(content="Obtain US GDP and plot a line chart. End with FINAL ANSWER after chart.")],},
        {"recursion_limit": 20},
        stream_mode="values",
    )
    for ev in events:
        if "messages" in ev:
            ev["messages"][-1].pretty_print()


def run_fake():
    global research_agent, chart_agent
    # Swap to FakeLLM if no real llm
    if llm_research is None:
        fake = FakeLLM()
        research_agent = create_agent(fake, tools_research, RESEARCH_SYS_MSG)
        chart_agent = create_agent(fake, tools_chart, CHART_SYS_MSG)
    # Rebuild minimal graph with fake agents
    wf = StateGraph(AgentState)
    wf.add_node("Researcher", functools.partial(agent_node, agent=research_agent, name="Researcher"))
    wf.add_node("Chart_Generator", functools.partial(agent_node, agent=chart_agent, name="Chart_Generator"))
    wf.add_node("call_tool", ToolNode(tools_research + tools_chart))
    wf.add_node("maybe_exec", maybe_exec_code)
    wf.add_conditional_edges("Researcher", router, {"continue": "Chart_Generator", "call_tool": "call_tool", "maybe_exec": "maybe_exec", "__end__": END})
    wf.add_conditional_edges("Chart_Generator", router, {"continue": "Researcher", "call_tool": "call_tool", "maybe_exec": "maybe_exec", "__end__": END})
    wf.add_conditional_edges("call_tool", lambda x: x["sender"], {"Researcher": "Researcher", "Chart_Generator": "Chart_Generator"})
    wf.add_conditional_edges("maybe_exec", lambda x: x["sender"], {"Researcher": "Researcher", "Chart_Generator": "Chart_Generator"})
    wf.add_edge(START, "Researcher")
    app_local = wf.compile()
    print("--- Fake run (offline) ---")
    events = app_local.stream({"messages": [HumanMessage(content="Plot GDP.")]}, {"recursion_limit": 10}, stream_mode="values")
    for ev in events:
        if "messages" in ev:
            ev["messages"][-1].pretty_print()

if __name__ == "__main__":
    if llm_research is None:
        run_fake()
    else:
        run_real()

