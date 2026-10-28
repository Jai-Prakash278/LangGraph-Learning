import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

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
# MCP SERVER
# ==========================================================
client = MultiServerMCPClient(
     {
        "ArithmeticServer" : {
             "transport" : "stdio",
             "command" : r"E:\MCP\MathServer\.venv\Scripts\python.exe",
             "args" : [r"E:\MCP\MathServer\src\mathserver\main.py"],
        },
        "ExpenseTracker": {
            "transport": "streamable_http",
            "url": "https://expense-tracker-server-hxcy.onrender.com/mcp",
        }
     }
)

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

async def build_graph():
    tools = await client.get_tools()
    print(tools)
    llm_with_tools = gemini_model.bind_tools(tools)

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

    chatbot = graph.compile()
    return chatbot

# ==========================================================
# MAIN
# ==========================================================

async def main():
    chatbot = await build_graph()
    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Add an expense of RS 8000 for delhi trip on 27 Sep"
                )
            ]
        },
    )
    print(
        result["messages"][-1].content
    )

if __name__ == "__main__":
    asyncio.run(main())