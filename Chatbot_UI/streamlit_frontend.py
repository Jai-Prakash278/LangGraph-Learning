import streamlit as st
from PIL import Image
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict ->
Config = {'configurable' : {'thread_id' : '1'}}

user_avatar = Image.open("assets/user2.jpg")
bot_avatar = Image.open("assets/bot.avif")

if 'message_history' not in st.session_state: 
    st.session_state['message_history'] = []

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

    response = chatbot.invoke({"messages" : [HumanMessage(content=user_input)]}, config=Config)

    ai_message = response['messages'][-1].content

    # First add the message to the message_history
    st.session_state['message_history'].append({'role' : 'assistant', 'content' : ai_message})
    with st.chat_message('assistant', avatar=bot_avatar):
        st.text(ai_message) 