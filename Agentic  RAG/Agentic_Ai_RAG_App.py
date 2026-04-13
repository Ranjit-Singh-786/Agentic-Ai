from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage
from typing import TypedDict
from dotenv import load_dotenv
import warnings
import json

load_dotenv()
warnings.filterwarnings("ignore")

llm = ChatOpenAI(model="gpt-4o-mini")


class RagState(TypedDict):
    query: str
    source_type: str
    source: str
    chunk_id: str
    chunk_length: int
    content: str
    last_question: str
    last_answer: str
    answeragent: str
    retrieval_output: str
    retrieval_fail_count: int


CHROMA_DIR = "./chroma_sbert_db"
COLLECTION_NAME = "pdf_url_sbert_collection"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)

print(f"Connected! Total chunks: {vectorstore._collection.count()}")


@tool
def RetrievalTool(query: str) -> dict:
    """This tool MUST be used for every user query to retrieve relevant context
    from the ChromaDB vector database. Do NOT answer any question without
    calling this tool first.
    """
    results = vectorstore.similarity_search(query, k=2)

    if results:
        doc = results[0]
        source_type = doc.metadata.get("source_type")
        source  = doc.metadata.get("source")
        chunk_id  = doc.metadata.get("chunk_id")
        chunk_length = doc.metadata.get("chunk_length", len(doc.page_content))
        page_content = doc.page_content[:200]

        return {
            "source_type":  source_type,
            "source":   source,
            "chunk_id":   chunk_id,
            "chunk_length":  chunk_length,
            "content":  page_content,
            "retrieval_output": page_content,
        }

    return {
        "source_type":  "unknown",
        "source":   "unknown",
        "chunk_id": "unknown",
        "chunk_length": 0,
        "content": "",
        "retrieval_output": "",
    }


tools = [RetrievalTool]
react_agent = create_react_agent(
    model=llm,
    tools=tools,
)


def RetrievalAgent(state: RagState):
    query = state["query"]
    previous_fail_count = state.get("retrieval_fail_count", 0)

    # LOGIC 2: 2 baar fail ho chuka hai to retrieval rok do
    if previous_fail_count >= 2:
        return {
            "source_type": "unknown",
            "source": "unknown",
            "chunk_id":  "unknown",
            "chunk_length": 0,
            "content":  "",
            "retrieval_output": "",
            "retrieval_fail_count": previous_fail_count,
        }

    agent_result = react_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    tool_result = None
    for message in agent_result.get("messages", []):
        if isinstance(message, ToolMessage):
            try:
                parsed = json.loads(message.content)
                if isinstance(parsed, dict) and "retrieval_output" in parsed:
                    tool_result = parsed
                    break
            except (json.JSONDecodeError, TypeError):
                pass

    if tool_result and tool_result.get("retrieval_output", ""):
        return {
            "source_type":  tool_result.get("source_type", "unknown"),
            "source":  tool_result.get("source", "unknown"),
            "chunk_id":  tool_result.get("chunk_id", "unknown"),
            "chunk_length": tool_result.get("chunk_length", 0),
            "content": tool_result.get("content", ""),
            "retrieval_output": tool_result.get("retrieval_output", ""),
            "retrieval_fail_count": 0,
        }

    # LOGIC 2: Retrieval fail — counter increment
    return {
        "source_type": "unknown",
        "source":  "unknown",
        "chunk_id":  "unknown",
        "chunk_length":  0,
        "content":  "",
        "retrieval_output": "",
        "retrieval_fail_count": previous_fail_count + 1,
    }


def AnswerAgent(state: RagState):
    query  = state["query"]
    last_question  = state.get("last_question", "")
    last_answer  = state.get("last_answer", "")
    retrieved_content = state.get("retrieval_output", "")
    fail_count  = state.get("retrieval_fail_count", 0)

    # LOGIC 2: 2 baar fail — hard fallback, no LLM call needed
    if fail_count >= 2:
        return {
            "answeragent": (
                "I'm sorry, I was unable to retrieve relevant information from the "
                "knowledge base after multiple attempts. Please try rephrasing your "
                "question or ask something else."
            )
        }

    prompt_text = f"""
You are a retrieval-grounded assistant.

Follow these rules strictly:
1. If the current query is a greeting (e.g. hi, hello, hey, good morning, etc.),
   respond warmly and politely, and ask what the user would like to know.
   Do not use retrieved context for greetings.

2. If the current query is the same as or very similar to the Last Question,
   politely acknowledge it and repeat the Last Answer as-is. Say something like:
   "You already asked this earlier. Here is the same answer:" followed by the answer.
   Do not re-derive or modify the answer.

3. Answer the current query using only the retrieved context.

4. Use the last conversation only for continuity, never as the primary factual source.

5. If the answer is not clearly supported by the retrieved context, respond with exactly:
   "I don't have enough information in the knowledge base to answer."

6. Do not invent, assume, or use outside knowledge.

7. Keep the answer concise, direct, and relevant to the query.

8. If your answer fully resolves the user's query based on the retrieved context,
   add this line at the end:
   "If you need anything else, feel free to ask. Otherwise, I'll end the session."
   If the answer is incomplete or unsupported by context, do NOT add this line.

Current Query:
{query}

Last Conversation:
Last Question: {last_question if last_question else "No previous question"}
Last Answer: {last_answer if last_answer else "No previous answer"}

Retrieved Context:
{retrieved_content if retrieved_content else "No retrieved context found"}
"""

    response = llm.invoke(prompt_text)
    return {"answeragent": response.content}


# Graph build
builder = StateGraph(RagState)
builder.add_node("retrieval_agent", RetrievalAgent)
builder.add_node("answer_agent", AnswerAgent)

builder.add_edge(START, "retrieval_agent")
builder.add_edge("retrieval_agent", "answer_agent")
builder.add_edge("answer_agent", END)

workflow = builder.compile()

# Conversation memory
conversation_memory = {
    "last_question":  "",
    "last_answer":  "",
    "last_source_type": "",
    "last_source":  "",
    "last_chunk_id": "",
    "last_chunk_length":  0,
    "retrieval_fail_count": 0,
}

while True:
    query = input("Enter your query  : ")

    if query.lower() in ["thanks", "good", "bye", "done", "okay"]:
        print("Thank You, Nice to meet you!")
        break

    initial_state = {
        "query":  query,
        "last_question":  conversation_memory["last_question"],
        "last_answer":   conversation_memory["last_answer"],
        "retrieval_fail_count": conversation_memory["retrieval_fail_count"],
    }

    final_state = workflow.invoke(initial_state)

    print("User       : ", query)
    print("Assistant  : ", final_state["answeragent"])
    print(
        f"Metadata   :\n"
        f"  Source Type  : {final_state['source_type']}\n"
        f"  Source       : {final_state['source']}\n"
        f"  Chunk ID     : {final_state['chunk_id']}\n"
        f"  Chunk Length : {final_state['chunk_length']}"
    )

    # LOGIC 2: Fail count update
    new_fail_count = final_state.get("retrieval_fail_count", 0)
    conversation_memory["retrieval_fail_count"] = new_fail_count

    # LOGIC 2: 2 baar fail — session end
    if new_fail_count >= 2:
        print("\n Retrieval has failed twice. Ending session.\n")
        break

    conversation_memory["last_question"]     = query
    conversation_memory["last_answer"]       = final_state["answeragent"]
    conversation_memory["last_source_type"]  = final_state["source_type"]
    conversation_memory["last_source"]       = final_state["source"]
    conversation_memory["last_chunk_id"]     = final_state["chunk_id"]
    conversation_memory["last_chunk_length"] = final_state["chunk_length"]

    print("\n" * 2)
