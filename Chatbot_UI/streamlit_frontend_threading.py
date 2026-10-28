import streamlit as st
from PIL import Image
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

user_avatar = Image.open("assets/user.jpg")
bot_avatar = Image.open("assets/bot.avif")


# ******************************** Utility Functions *********************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_conversation(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config=config)

    return state.values["messages"]        

# ******************************** Session Setup **************************************
if 'message_history' not in st.session_state: 
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

# ******************************* Sidebar UI *****************************************

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversation')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'

            temp_messages.append({'role' : role, 'content' : msg.content})

        st.session_state['message_history'] = temp_messages

# ******************************* Main UI ********************************************

# Loading the conversation history
for message in st.session_state['message_history'] :

    avatar = user_avatar if message["role"] == "user" else bot_avatar

    with st.chat_message(message['role'], avatar=avatar):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:

    # First add the message to the message_history
    st.session_state['message_history'].append({'role' : 'user', 'content' : user_input})
    with st.chat_message('user', avatar=user_avatar):
        st.text(user_input)

    Config = {'configurable' : {'thread_id' : st.session_state['thread_id']}}

    # First add the message to the message_history
    with st.chat_message('assistant', avatar=bot_avatar):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages' : [HumanMessage(content=user_input)]},
                config=Config,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role' : 'assistant', 'content' : ai_message})