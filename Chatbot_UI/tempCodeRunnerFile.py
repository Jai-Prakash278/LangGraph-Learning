import asyncio
import sqlite3
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


load_dotenv()


# ==========================================================
# DATABASE
# ==========================================================

connection = sqlite3.connect(
    "database.db",
    check_same_thread=False,
)

cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

connection.commit()

checkpointer = SqliteSaver(connection)

# ==========================================================
# MODELS
# ==========================================================

groq_model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# ==========================================================
# TOOLS
# ==========================================================

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """Perform arithmetic operations."""

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero"}

        result = first_num / second_num

    elif operation == "mod":
        if second_num == 0:
            return {"error": "Division by zero"}

        result = first_num % second_num

    else:
        return {
            "error": f"Unsupported operation: {operation}"
        }

    return {
        "first_number": first_num,
        "second_number": second_num,
        "operation": operation,
        "result": result,
    }

tools = [calculator]

llm_with_tools = groq_model.bind_tools(tools)

# ==========================================================
# STATE
# ==========================================================

class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

# ==========================================================
# GRAPH
# ==========================================================

def build_graph():
    async def chat_node(state: ChatState):
            """LLM node that may answer or request a tool call."""
            messages = state["messages"]
            response = await llm_with_tools.ainvoke(messages)
            return {
                "messages": [response]
            }
    
    tool_node = ToolNode(tools)  #Executes tool calls
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat")
    graph.add_conditional_edges(
        "chat",
        tools_condition,
    )
    graph.add_edge(
        "tools",
        "chat",
    )
    return graph.compile(
        checkpointer=checkpointer
    )

# ==========================================================
# MAIN
# ==========================================================

async def main():
    chatbot = build_graph()
    config = {
        "configurable": {
            "thread_id": "test-thread-1"
        }
    }
    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Find the modulus of "
                        "132354 and 24 and explain "
                        "it like a cricket commentator."
                    )
                )
            ]
        },
        config=config,
    )
    print(
        result["messages"][-1].content
    )

if __name__ == "__main__":
    asyncio.run(main())