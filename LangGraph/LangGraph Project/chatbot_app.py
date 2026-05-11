import streamlit as st 
from workflow import Workflow
st.header("ChatBot With LangGraph workflow")

config  = {"configurable":{"thread_id":"user-123"}}

if "message_history" not in st.session_state:
    st.session_state["message_history"] = [] 

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]) as box:
        st.text(message['content'])


user_input = st.chat_input("Type here ...")

if user_input:
    st.session_state['message_history'].append({"role":"user","content":user_input})
    with st.chat_message("user") as box:
        st.text(user_input)

    result = Workflow.invoke({"messages":user_input},config=config)
    st.session_state['message_history'].append({"role":"assistant","content":result['messages'][-1].content})
    with st.chat_message("assistant") as box:
        st.text(result['messages'][-1].content)
