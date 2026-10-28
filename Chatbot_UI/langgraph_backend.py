from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

chat_model1 = ChatGroq(
    model="llama-3.3-70b-versatile"
)

class ChatState(TypedDict) :
    messages : Annotated[list[BaseMessage], add_messages]

def chat_func(state : ChatState) :
    message = state["messages"]
    response = chat_model1.invoke(message)
    return {'messages' : [response]}

Checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node('Chat', chat_func)

graph.add_edge(START, 'Chat')
graph.add_edge('Chat', END)

chatbot = graph.compile(checkpointer=Checkpointer)