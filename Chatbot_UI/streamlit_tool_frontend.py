import streamlit as st
from PIL import Image
from langgraph_tool_backend import (
    chatbot,
    retrieval_all_threads,
    generate_title,
    create_conversation,
    update_conversation_title,
    get_conversation_title,
    delete_conversation
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ============================================
# Images
# ============================================

user_avatar = Image.open("assets/user.jpg")
bot_avatar = Image.open("assets/bot.avif")

# ============================================
# Utility Functions
# ============================================

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    create_conversation(thread_id)
    st.rerun()

def delete_chat(thread_id):
    delete_conversation(thread_id)

    # If deleting the currently open chat
    if st.session_state["thread_id"] == thread_id:
        new_thread_id = generate_thread_id()
        st.session_state["thread_id"] = new_thread_id
        st.session_state["message_history"] = []
        create_conversation(new_thread_id)
    st.rerun()

def load_conversation(thread_id):

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    state = chatbot.get_state(config=config)
    if state is None:
        return []
    messages = state.values.get("messages", [])
    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({
                "role": "user",
                "content": msg.content
            })
        elif isinstance(msg, AIMessage):
            # Ignore AI messages that only contain tool calls
            if msg.tool_calls:
                continue
            if isinstance(msg.content, str) and msg.content.strip():
                temp_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
        # IMPORTANT:
        # Do NOT display ToolMessage
    return temp_messages


# ============================================
# Session
# ============================================

if "thread_id" not in st.session_state:
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    create_conversation(thread_id)

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# ============================================
# Sidebar
# ============================================

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

chat_threads = retrieval_all_threads()

for thread_id, title in chat_threads:
    col1, col2 = st.sidebar.columns([5, 1])
    with col1:
        if st.button(
            title,
            key=f"chat_{thread_id}"
        ):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(
                thread_id
            )
            st.rerun()
    with col2:
        if st.button(
            "🗑️",
            key=f"delete_{thread_id}"
        ):
            delete_chat(thread_id)

# ============================================
# Chat Window
# ============================================

for message in st.session_state["message_history"]:
    avatar = (
        user_avatar
        if message["role"] == "user"
        else bot_avatar
    )
    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.write(message["content"])

# ============================================
# User Input
# ============================================

user_input = st.chat_input("Type here...")

if user_input:

    # ----------------------------------------
    # USER MESSAGE
    # ----------------------------------------

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message(
        "user",
        avatar=user_avatar
    ):
        st.write(user_input)
    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata" : {
            "thread_id" : st.session_state["thread_id"]
        },
        "run_name" : "Chat_Turn"
    }

    # ----------------------------------------
    # AI RESPONSE
    # ----------------------------------------

    def ai_only_stream():
        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            # Only receive output from Chat_Node
            if metadata.get("langgraph_node") != "Chat_Node":
                continue
            # Only AI messages
            if not isinstance(message_chunk, AIMessage):
                continue
            # Ignore tool-calling messages
            if message_chunk.tool_calls:
                continue
            content = message_chunk.content
            if isinstance(content, str) and content.strip():
                yield content

    with st.chat_message(
        "assistant",
        avatar=bot_avatar
    ):
        ai_message = st.write_stream(
            ai_only_stream()
        )

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    # =====================================================
    # Generate title ONLY if it is still Untitled Chat
    # =====================================================

    current_title = get_conversation_title(
        st.session_state["thread_id"]
    )

    if current_title == "Untitled Chat":
        first_user = None
        first_ai = None
        for message in st.session_state["message_history"]:
            if (
                message["role"] == "user"
                and first_user is None
            ):
                first_user = message["content"]
            elif (
                message["role"] == "assistant"
                and first_ai is None
            ):
                first_ai = message["content"]
            if first_user and first_ai:
                break
        if first_user and first_ai:
            title = generate_title(
                first_user,
                first_ai
            )
            update_conversation_title(
                st.session_state["thread_id"],
                title
            )
            st.rerun()