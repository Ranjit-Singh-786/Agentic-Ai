import streamlit as st 
from workflow import Workflow
import warnings
import uuid 
warnings.filterwarnings("ignore")

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def add_thread_id(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


# -------------------- session_state setup  ----------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = [] 

if "thread_id" not in st.session_state:  # current conversation ID
    st.session_state["thread_id"] = generate_thread_id()
    add_thread_id(st.session_state['thread_id'])

# if "chat_thread_history" not in st.session_state:
#     st.session_state['chat_thread_history'] = st.session_state['message_history']


# ----------------------------------------------------------------


st.header("ChatBot With LangGraph workflow")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]) as box:
        st.text(message['content'])



# ------------------------------- SIDE BAR ------------------------------------

st.sidebar.title("Your Conversation")
if st.sidebar.button("New Chat"):
    add_thread_id(st.session_state['thread_id'])
    st.session_state['thread_id']=generate_thread_id()

for thread in st.session_state['chat_threads'][::-1]:
    st.sidebar.text(thread)
# -----------------------------------------------------------------------------




user_input = st.chat_input("Type here ...")

if user_input:
    st.session_state['message_history'].append({"role":"user","content":user_input})
    with st.chat_message("user") as box:
        st.text(user_input)
    config  = {"configurable":{"thread_id":st.session_state['thread_id']}}

    result = Workflow.invoke({"messages":user_input},config=config)
    st.session_state['message_history'].append({"role":"assistant","content":result['messages'][-1].content})
    with st.chat_message("assistant") as box:
        st.text(result['messages'][-1].content)
