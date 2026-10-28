from __future__ import annotations

import atexit
import os
import tempfile
import uuid
from typing import Annotated, Any, Dict, List, Optional, TypedDict

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

DB_URI = os.getenv("DB_URI")

if not DB_URI:
    raise ValueError(
        "DB_URI not found in .env. "
        "Please configure your PostgreSQL connection string."
    )

# ============================================================
# 2. LLM AND EMBEDDINGS
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.7,
)

memory_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)


# ============================================================
# 3. LONG-TERM MEMORY MODELS
# ============================================================

class MemoryItem(BaseModel):
    text: str = Field(
        description="One short, atomic user memory"
    )

    is_new: bool = Field(
        description="True only when this fact is genuinely new"
    )


class MemoryDecision(BaseModel):
    should_write: bool = Field(
        description="Whether new user memories should be saved"
    )

    memories: List[MemoryItem] = Field(
        default_factory=list
    )


# Define the models BEFORE creating the structured-output chain.
memory_extractor = memory_llm.with_structured_output(
    MemoryDecision
)


# ============================================================
# 4. MEMORY PROMPTS
# ============================================================

MEMORY_PROMPT = """
You maintain accurate long-term memory about the user.

EXISTING MEMORIES:
{user_details_content}

Review the latest user message and extract useful,
user-specific information worth remembering long-term.

Prioritize:
- Stable preferences
- Ongoing projects and goals
- Relevant skills and experience
- Important user-provided facts

Rules:
- Store only facts explicitly stated or clearly established.
- Never store passwords, API keys, tokens, or secrets.
- Avoid storing temporary information without future value.
- Do not duplicate existing memories.
- Mark is_new=true only when information is genuinely new.
- Keep each memory short, atomic, and understandable.
- Do not infer personal facts from generic questions.
- If nothing is worth storing, return should_write=false
  and an empty memories list.
"""


SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant with long-term memory capabilities.

Use the following memories when they are relevant to the user's
current request.

Do not invent facts or claim to remember information that is
not present in the memories or conversation.

USER MEMORIES:
{user_details_content}

Answer the current question directly.
Personalize responses when relevant.
"""


# ============================================================
# 5. PDF RETRIEVER STORAGE
# ============================================================
# Note: These retrievers are stored in application memory.
# They will not survive a backend process restart.

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Get the PDF retriever associated with a conversation."""
    if thread_id is None:
        return None
    return _THREAD_RETRIEVERS.get(str(thread_id))


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ============================================================
# 6. TOOLS
# ============================================================

_ddg_search = DuckDuckGoSearchRun(region="us-en")

@tool
def search_tool(query: str) -> str:
    """Search the web for current information."""
    try:
        return _ddg_search.invoke(query)
    except Exception as exc:
        return (
            "Web search is temporarily unavailable. "
            f"Search error: {exc}"
        )

@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform basic arithmetic.
    Supported operations: add, sub, mul, div.
    """
    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed."
                }

            result = first_num / second_num
        else:
            return {
                "error": (
                    f"Unsupported operation: {operation}. "
                    "Use add, sub, mul, or div."
                )
            }
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except (TypeError, ValueError, OverflowError) as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock quote using Alpha Vantage.
    Example symbols: AAPL, TSLA.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {"error": "ALPHA_VANTAGE_API_KEY is missing from .env"}

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.strip().upper(),
        "apikey": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data or "Information" in data:
            return {
                "error": data.get("Note") or data.get("Information")
            }

        quote = data.get("Global Quote", {})
        if not quote:
            return {"error": "No stock quote returned.", "response": data}

        return quote

    except requests.RequestException as exc:
        return {"error": f"Stock API request failed: {exc}"}



@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 7. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[
        List[BaseMessage],
        add_messages,
    ]

# ============================================================
# 8. LONG-TERM MEMORY NODE
# ============================================================

def remember_node(
    state: ChatState,
    config: RunnableConfig,
    store: BaseStore,
) -> dict:
    """
    Extract new facts from the latest user message and save
    them in PostgreSQL under that user's namespace.
    """
    configurable = (config or {}).get(
        "configurable", {}
    )

    user_id = configurable.get("user_id")

    if not user_id:
        raise ValueError(
            "configurable.user_id is required "
            "for long-term memory."
        )

    namespace = (
        "user",
        str(user_id),
        "details",
    )

    # Retrieve existing memories.
    existing_items = store.search(namespace)

    existing_memories = [
        str(item.value.get("data", "")).strip()
        for item in existing_items
        if item.value.get("data")
    ]

    existing_text = (
        "\n".join(existing_memories)
        if existing_memories
        else "(empty)"
    )

    # Find the latest human message.
    last_user_message = next(
        (
            message
            for message in reversed(state["messages"])
            if isinstance(message, HumanMessage)
        ),
        None,
    )

    if last_user_message is None:
        return {}

    latest_text = last_user_message.content

    # This implementation handles plain-text messages.
    if (
        not isinstance(latest_text, str)
        or not latest_text.strip()
    ):
        return {}

    # Ask Gemini to extract useful new memories.
    decision: MemoryDecision = memory_extractor.invoke(
        [
            SystemMessage(
                content=MEMORY_PROMPT.format(
                    user_details_content=existing_text
                )
            ),
            HumanMessage(
                content=latest_text.strip()
            ),
        ]
    )

    if not decision.should_write:
        return {}

    # Prevent exact duplicate entries.
    known_memories = {
        memory.casefold()
        for memory in existing_memories
        if memory
    }

    for memory in decision.memories:
        memory_text = memory.text.strip()

        if not memory.is_new or not memory_text:
            continue

        normalized_text = memory_text.casefold()

        if normalized_text in known_memories:
            continue

        store.put(
            namespace,
            str(uuid.uuid4()),
            {"data": memory_text},
        )

        known_memories.add(normalized_text)

    return {}


# ============================================================
# 9. CHAT NODE WITH MEMORY
# ============================================================

def chat_node(
    state: ChatState,
    config: RunnableConfig,
    *,
    store: BaseStore,
):
    """
    Load the user's long-term memories and generate a response
    using Gemini and the existing tools.
    """

    configurable = (config or {}).get(
        "configurable", {}
    )

    user_id = configurable.get("user_id")
    thread_id = configurable.get("thread_id")

    if not user_id:
        raise ValueError(
            "configurable.user_id is required "
            "for long-term memory."
        )

    namespace = (
        "user",
        str(user_id),
        "details",
    )

    # Retrieve memories from PostgreSQL.
    items = store.search(namespace)

    user_details = "\n".join(
        str(item.value.get("data", "")).strip()
        for item in items
        if item.value.get("data")
    )

    system_message = SystemMessage(
        content=(
            SYSTEM_PROMPT_TEMPLATE.format(
                user_details_content=(
                    user_details or "(empty)"
                )
            )
            + "\n\nFor questions about an uploaded PDF, "
            + "use rag_tool and pass the current thread_id "
            + f"{str(thread_id)!r}. "
            + "Use web search, calculator, and stock-price "
            + "tools when appropriate."
        )
    )

    response = llm_with_tools.invoke(
        [
            system_message,
            *state["messages"],
        ],
        config=config,
    )

    return {"messages": [response]}


# ============================================================
# POSTGRESQL CHECKPOINTER
# ============================================================

_postgres_checkpointer_context = (
    PostgresSaver.from_conn_string(DB_URI)
)

checkpointer = _postgres_checkpointer_context.__enter__()

# Initialize checkpoint tables and migrations.
checkpointer.setup()


# ============================================================
# 11. POSTGRESQL LONG-TERM STORE
# ============================================================
# Keep the store open while the graph is being used.
# Do not close it immediately after graph compilation.

_postgres_store_context = (
    PostgresStore.from_conn_string(DB_URI)
)

store = _postgres_store_context.__enter__()

# Initialize PostgreSQL store tables/indexes.
# Run this during initial setup/application startup.
store.setup()


# ============================================================
# 12. BUILD THE LANGGRAPH WORKFLOW
# ============================================================

builder = StateGraph(ChatState)

builder.add_node(
    "remember",
    remember_node,
)

builder.add_node(
    "chat_node",
    chat_node,
)

builder.add_node(
    "tools",
    ToolNode(tools),
)

# Save memories before generating the response.
builder.add_edge(
    START,
    "remember",
)

builder.add_edge(
    "remember",
    "chat_node",
)

# If Gemini requests a tool, execute it.
# Otherwise, finish the graph.
builder.add_conditional_edges(
    "chat_node",
    tools_condition,
)

# Return tool results to the chat node.
builder.add_edge(
    "tools",
    "chat_node",
)

# Compile with BOTH persistence mechanisms.
chatbot = builder.compile(
    checkpointer=checkpointer,
    store=store,
)


# ============================================================
# 13. HELPER FUNCTIONS
# ============================================================

def retrieve_all_threads() -> List[str]:
    """Return conversation thread IDs from PostgreSQL checkpoints."""

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        configurable = checkpoint.config.get(
            "configurable", {}
        )

        thread_id = configurable.get("thread_id")

        if thread_id:
            all_threads.add(str(thread_id))

    return list(all_threads)


def thread_has_document(
    thread_id: str,
) -> bool:
    """Check whether a PDF retriever is loaded for a thread."""

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(
    thread_id: str,
) -> dict:
    """Return metadata for a thread's uploaded PDF."""

    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )


def get_user_memories(
    user_id: str,
) -> List[str]:
    """Return stored long-term memories for a user."""

    namespace = (
        "user",
        str(user_id),
        "details",
    )

    return [
        str(item.value.get("data", "")).strip()
        for item in store.search(namespace)
        if item.value.get("data")
    ]


# ============================================================
# 14. RESOURCE CLEANUP
# ============================================================

def close_backend_resources() -> None:
    """Close database resources when the process exits."""

    try:
        _postgres_store_context.__exit__(
            None,
            None,
            None,
        )
    except Exception:
        pass


atexit.register(close_backend_resources)


# ============================================================
# 15. OPTIONAL STANDALONE TEST
# ============================================================
# Uncomment this block to test the backend directly.
# Do not run it automatically when importing this module
# from your frontend or API server.

# if __name__ == "__main__":

#     config = {
#         "configurable": {
#             "thread_id": "ltm-test-thread-1",
#             "user_id": "ltm-test-user-1",
#         }
#     }

#     result = chatbot.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content=(
#                         "what is the stock price of AAPL?"
#                     )
#                 )
#             ]
#         },
#         config=config,
#     )

#     print(result["messages"][-1].content)

#     print("\nStored memories:")

#     for memory in get_user_memories("ltm-test-user-1"):
#         print("-", memory)



# if __name__ == "__main__":
#     print(get_stock_price.invoke({"symbol": "AAPL"}))