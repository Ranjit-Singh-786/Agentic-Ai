from langgraph.graph import StateGraph,END ,START 
from langchain_core.messages import BaseMessage 
from langchain_openai import ChatOpenAI 
from typing import TypedDict,Annotated 
from langgraph.graph.message import add_messages 
from langgraph.checkpoint.memory import MemorySaver 
from dotenv import load_dotenv
load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]


llm = ChatOpenAI(model="gpt-4o-mini") 
def ChatNode(state:ChatState)->dict:
    message = state["messages"]
    response = llm.invoke(message)
    return {"messages":[response]}

checkpointer = MemorySaver()
graph = StateGraph(ChatState) 
graph.add_node("chat_node",ChatNode)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END )
Workflow = graph.compile(checkpointer=checkpointer)