from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

# ===========================
# LLMs
# ===========================

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chat_model1 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# ===========================
# LangGraph State
# ===========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ===========================
# Chat Node
# ===========================

def chat_func(state: ChatState):

    messages = state["messages"]

    response = chat_model1.invoke(messages)

    return {
        "messages": [response]
    }

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

graph = StateGraph(ChatState)

graph.add_node("Chat", chat_func)

graph.add_edge(START, "Chat")
graph.add_edge("Chat", END)

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