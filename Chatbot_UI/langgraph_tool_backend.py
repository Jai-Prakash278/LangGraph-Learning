from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import sqlite3
import requests

load_dotenv()

# ===========================
# LLMs
# ===========================

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chat_model1 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
)

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num : float, second_num : float, operation : str) -> dict :
    """
    Perform a basic arithmetic operation on two numbers.
    Support operations : add, sub, mul, div
    """
    try : 
        if operation == 'add' :
            result = first_num + second_num
        elif operation == 'sub' :
            result = first_num - second_num
        elif operation == 'mul' :
            result = first_num * second_num
        elif operation == 'div' :
            if second_num == 0 :
                return {"error" : "Division by zero is not allowed"}
            result = first_num/second_num
        else : 
            return {"error" : f"Unsupported operation '{operation}'"}

        return {"first number" : first_num, "Second number" : second_num, "Operation" : operation, "Result" : result}
    except Exception as e :
        return {"error" : str(e)}

@tool
def get_stock_price(symbol : str) -> dict :
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key in the URL.
    """
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&interval=5min&apikey= WOOGZ31BEOPHGT6T'
    r = requests.get(url)
    return r.json()


# Make tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = chat_model1.bind_tools(tools)

# ===========================
# LangGraph State
# ===========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===========================
# Chat Node
# ===========================

def chat_func(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response]
    }

tool_node = ToolNode(tools)  #Executes tool calls

# ===========================
# Database
# ===========================

connection = sqlite3.connect(
    "database.db",
    check_same_thread=False
)
cursor = connection.cursor()

# Conversation Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations(
    thread_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
connection.commit()

# ===========================
# LangGraph Checkpointer
# ===========================

Checkpointer = SqliteSaver(connection)

# Graph Structure
graph = StateGraph(ChatState)
# Nodes
graph.add_node('Chat_Node', chat_func)
graph.add_node('tools', tool_node)
# Edges
graph.add_edge(START, 'Chat_Node')
# if the LLM asked for a tool, go to ToolNode; else : finish
graph.add_conditional_edges('Chat_Node', tools_condition)
graph.add_edge('tools', 'Chat_Node')

chatbot = graph.compile(
    checkpointer=Checkpointer
)

# ==========================================================
# Conversation Functions
# ==========================================================

def retrieval_all_threads():
    cursor = connection.cursor()
    cursor.execute("""
        SELECT thread_id,title
        FROM conversations
        ORDER BY updated_at DESC
    """)
    return cursor.fetchall()

def create_conversation(thread_id):
    cursor = connection.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO conversations(
            thread_id,
            title
        )
        VALUES (?,?)
    """, (
        str(thread_id),
        "Untitled Chat"
    ))
    connection.commit()

def delete_conversation(thread_id):
    thread_id = str(thread_id)
    # Delete LangGraph checkpoint/history
    Checkpointer.delete_thread(thread_id)
    # Delete conversation metadata
    cursor = connection.cursor()
    cursor.execute(
        """
        DELETE FROM conversations
        WHERE thread_id = ?
        """,
        (thread_id,)
    )
    connection.commit()

def update_conversation_title(thread_id, title):
    cursor = connection.cursor()
    cursor.execute("""
        UPDATE conversations
        SET
            title=?,
            updated_at=CURRENT_TIMESTAMP
        WHERE thread_id=?
    """, (
        title,
        str(thread_id)
    ))
    connection.commit()

def get_conversation_title(thread_id):
    cursor = connection.cursor()
    cursor.execute("""
        SELECT title
        FROM conversations
        WHERE thread_id=?
    """, (
        str(thread_id),
    ))
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

# ==========================================================
# AI Title Generator
# ==========================================================

def generate_title(
    user_message,
    assistant_message
):
    prompt = f"""
You generate conversation titles.
Rules:
- Maximum 5 words
- Title Case
- No quotation marks
- No punctuation
Return ONLY the title.
Conversation
User:
{user_message}

Assistant:
{assistant_message}
"""
    response = chat_model.invoke(prompt)
    return response.content.strip()