"""Multi-agent collaboration with a Table_Generator agent (LangChain 0.2.x + LangGraph 0.0.69).
Replaces the previous Chart_Generator with a Table_Generator that produces tabular data using Python (pandas).
Logic mirrors the chart version: researcher supplies context, table generator writes Python code, MUST execute it before FINAL ANSWER.
Fallback node auto-executes code blocks without explicit tool_calls.

Run:
    python multi_agent_collaboration_table.py

Prereqs:
    - Environment set up with downgraded LangChain 0.2.x + langgraph==0.0.69
    - OPENAI / Azure environment vars if using real LLM; else FakeLLM will simulate.
"""
from __future__ import annotations
import os, re, functools
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

# -----------------------------------------------------------------------------
# LLM setup (attempt AzureChatOpenAI first, fallback to OpenAI, else FakeLLM)
# -----------------------------------------------------------------------------
try:
    import azure_chat_openai
    llm_research = azure_chat_openai.chat_model
    llm_table = azure_chat_openai.chat_model
except Exception:
    try:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm_research = ChatOpenAI(model=model_name, temperature=0)
        llm_table = llm_research
    except Exception:
        llm_research = llm_table = None  # Use FakeLLM below

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------
python_env = PythonREPL()

@tool
def python_repl(code: Annotated[str, "Python code that generates a pandas DataFrame or table text."]):
    """Execute Python code and return stdout plus any DataFrame markdown.
    If a variable named `df` exists and is a pandas DataFrame, include its markdown form.
    """
    import io, base64
    try:
        result = python_env.run(code)
        # Attempt to capture a 'df' variable if user created it.
        # PythonREPL stores variables inside python_env.locals
        df_markdown = ""
        if "df" in python_env.locals:
            try:
                import pandas as pd
                if isinstance(python_env.locals["df"], pd.DataFrame):
                    df_markdown = "\nDATAFRAME_MARKDOWN:\n" + python_env.locals["df"].to_markdown(index=False)
            except Exception as e:
                df_markdown = f"\nFailed to render DataFrame markdown: {e!r}"
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return (
        f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}{df_markdown}\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )

@tool
def dummy_search(query: Annotated[str, "Search query covering all aspects."]):
    return f"(dummy search results for: {query})"

research_tools = [dummy_search]
table_tools = [python_repl]

# -----------------------------------------------------------------------------
# Helper: create_agent
# -----------------------------------------------------------------------------

def create_agent(llm, tools, system_message: str):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful AI assistant collaborating with other assistants. "
            "Use provided tools. If final deliverable ready, prefix with FINAL ANSWER. Tools: {tool_names}.\n{system_message}",
        ),
        MessagesPlaceholder("messages"),
    ]).partial(system_message=system_message).partial(tool_names=", ".join([t.name for t in tools]))
    if llm is None:
        return prompt | FakeLLM()  # for offline tests
    return prompt | llm.bind_tools(tools)

RESEARCH_SYS = (
    "Clarify the data needed (years, GDP values). Provide a concise structured list or CSV snippet for the table generator. "
    "Perform one efficient search (dummy tool) if needed before handing off."
)
TABLE_SYS = (
    "Generate a clean tabular representation of the provided data. Always write Python code that constructs the table (e.g., a pandas DataFrame named df). "
    "You MUST call the python_repl tool with the code before giving FINAL ANSWER. If you output a code block without tool call it will be auto executed."
)

research_agent = create_agent(llm_research, research_tools, RESEARCH_SYS)
table_agent = create_agent(llm_table, table_tools, TABLE_SYS)

# -----------------------------------------------------------------------------
# Agent node wrapper
# -----------------------------------------------------------------------------

def agent_node(state, agent, name):
    result = agent.invoke(state)
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {"messages": [result], "sender": name}

research_node = functools.partial(agent_node, agent=research_agent, name="Researcher")
table_node = functools.partial(agent_node, agent=table_agent, name="Table_Generator")

# -----------------------------------------------------------------------------
# State schema
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], list]
    sender: str

# -----------------------------------------------------------------------------
# Fallback execution node (auto run bare code fence for Table_Generator)
# -----------------------------------------------------------------------------

def maybe_exec_code(state: AgentState):
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if state.get("sender") != "Table_Generator":
        return {"messages": [], "sender": state["sender"]}
    if "Successfully executed:" in content:
        return {"messages": [], "sender": state["sender"]}
    if "```python" in content and not getattr(last, "tool_calls", None):
        m = re.search(r"```python\n(.*?)```", content, re.DOTALL)
        if m:
            code = m.group(1)
            try:
                exec_result = python_env.run(code)
                df_markdown = ""
                if "df" in python_env.locals:
                    import pandas as pd
                    if isinstance(python_env.locals["df"], pd.DataFrame):
                        df_markdown = "\nDATAFRAME_MARKDOWN:\n" + python_env.locals["df"].to_markdown(index=False)
            except Exception as e:
                exec_result = f"Tool error: {e!r}"
                df_markdown = ""
            tool_msg = ToolMessage(
                content=(
                    f"Successfully executed:\n```python\n{code}\n```\nStdout: {exec_result}{df_markdown}\n\nFINAL ANSWER"
                ),
                name="python_repl",
                tool_call_id="manual_python",
            )
            return {"messages": [tool_msg], "sender": state["sender"]}
    return {"messages": [], "sender": state["sender"]}

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------

def router(state: AgentState) -> Literal["call_tool", "__end__", "continue", "maybe_exec"]:
    last = state["messages"][-1]
    content = getattr(last, "content", "") or ""
    if getattr(last, "tool_calls", None):
        return "call_tool"
    if state.get("sender") == "Table_Generator" and "```python" in content and "Successfully executed:" not in content:
        return "maybe_exec"
    if "FINAL ANSWER" in content:
        return "__end__"
    return "continue"

# -----------------------------------------------------------------------------
# Graph build
# -----------------------------------------------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("Researcher", research_node)
workflow.add_node("Table_Generator", table_node)
workflow.add_node("call_tool", ToolNode(research_tools + table_tools))
workflow.add_node("maybe_exec", maybe_exec_code)

workflow.add_conditional_edges(
    "Researcher",
    router,
    {
        "continue": "Table_Generator",
        "call_tool": "call_tool",
        "maybe_exec": "maybe_exec",
        "__end__": END,
    },
)
workflow.add_conditional_edges(
    "Table_Generator",
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
    {"Researcher": "Researcher", "Table_Generator": "Table_Generator"},
)
workflow.add_conditional_edges(
    "maybe_exec",
    lambda x: x["sender"],
    {"Researcher": "Researcher", "Table_Generator": "Table_Generator"},
)
workflow.add_edge(START, "Researcher")
app = workflow.compile()

# -----------------------------------------------------------------------------
# FakeLLM for offline validation if no real llm
# -----------------------------------------------------------------------------
class FakeLLM:
    def __init__(self):
        self.step = 0
    def bind_tools(self, tools):
        return self
    def invoke(self, state):
        sender = state.get("sender")
        if sender == "Researcher":
            return AIMessage(content="Years,GDP\n2000,10284.8\n2001,10621.8\n2002,10978.2", name="Researcher")
        if sender == "Table_Generator":
            if self.step == 0:
                self.step += 1
                code = (
                    "```python\nimport pandas as pd\nrows=[(2000,10284.8),(2001,10621.8),(2002,10978.2)]\n" \
                    "df=pd.DataFrame(rows, columns=['Year','GDP'])\nprint(df)\n```\n(Will execute next)"
                )
                return AIMessage(content=code, name="Table_Generator")
            return AIMessage(content="FINAL ANSWER: Table generated.", name="Table_Generator")
        return AIMessage(content="Unexpected sender", name=sender)

if llm_research is None:
    # Replace agents with fake ones
    fake = FakeLLM()
    research_agent = create_agent(fake, research_tools, RESEARCH_SYS)
    table_agent = create_agent(fake, table_tools, TABLE_SYS)

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def run(task: str):
    events = app.stream(
        {"messages": [HumanMessage(content=task)]}, {"recursion_limit": 20}, stream_mode="values"
    )
    seen_exec = False
    for ev in events:
        if "messages" in ev:
            msg = ev["messages"][-1]
            if isinstance(msg, ToolMessage) and "Successfully executed:" in msg.content:
                seen_exec = True
            # Display DataFrame markdown if present
            if isinstance(msg, ToolMessage) and "DATAFRAME_MARKDOWN:" in msg.content:
                # Extract markdown after marker
                md = msg.content.split("DATAFRAME_MARKDOWN:",1)[1].strip()
                print("\n--- DataFrame (markdown) ---\n" + md + "\n----------------------------\n")
            msg.pretty_print()
    assert seen_exec, "Table code was not executed via tool."

if __name__ == "__main__":
    run("Obtain US GDP for 2000-2002 and build a table. End after table generation.")

