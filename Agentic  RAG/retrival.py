from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv
import warnings

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


def normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def query_data(query: str, k: int = 2):
    print(f"\nQuery: {query}")

    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\nResult {i + 1}")
        print(f"Source Type : {doc.metadata.get('source_type')}")
        print(f"Source      : {doc.metadata.get('source')}")
        print(f"Chunk ID    : {doc.metadata.get('chunk_id')}")
        print(f"Chunk Length: {doc.metadata.get('chunk_length', len(doc.page_content))}")
        print(f"Content     : {doc.page_content[:200]}...")


def RetrievalAgent(state: RagState):
    query = state["query"]
    results = vectorstore.similarity_search(query, k=1)

    if results:
        doc = results[0]
        source_type = doc.metadata.get("source_type")
        source = doc.metadata.get("source")
        chunk_id = doc.metadata.get("chunk_id")
        chunk_length = doc.metadata.get("chunk_length", len(doc.page_content))
        page_content = doc.page_content[:200]

        return {
            "source_type": source_type,
            "source": source,
            "chunk_id": chunk_id,
            "chunk_length": chunk_length,
            "content": page_content,
            "retrieval_output": page_content,
        }

    return {
        "source_type": "unknown",
        "source": "unknown",
        "chunk_id": "unknown",
        "chunk_length": 0,
        "content": "",
        "retrieval_output": "",
    }


def AnswerAgent(state: RagState):
    query = state["query"]
    last_question = state.get("last_question", "")
    last_answer = state.get("last_answer", "")
    retrieved_content = state["retrieval_output"]

    prompt = f"""
You are a retrieval-grounded assistant.

Follow these rules strictly:
1. Answer the current query using only the retrieved context.
2. Use the last conversation only for continuity, never as the primary factual source.
3. If the answer is not clearly supported by the retrieved context, respond with exactly:
"I don’t have enough information in the knowledge base to answer."
4. Do not invent, assume, or use outside knowledge.
5. Keep the answer concise, direct, and relevant to the query.

Current Query:
{query}

Last Conversation:
Last Question: {last_question if last_question else "No previous question"}
Last Answer: {last_answer if last_answer else "No previous answer"}

Retrieved Context:
{retrieved_content if retrieved_content else "No retrieved context found"}
"""

    response = llm.invoke(prompt)
    return {"answeragent": response.content}


builder = StateGraph(RagState)
builder.add_node("retrieval_agent", RetrievalAgent)
builder.add_node("answer_agent", AnswerAgent)

builder.add_edge(START, "retrieval_agent")
builder.add_edge("retrieval_agent", "answer_agent")
builder.add_edge("answer_agent", END)

workflow = builder.compile()

conversation_memory = {
    "last_question": "",
    "last_answer": "",
    "last_source_type": "",
    "last_source": "",
    "last_chunk_id": "",
    "last_chunk_length": 0,
}

while True:
    query = input("Enter your query  : ")
    if query.lower() in ["thanks", "good", "bye", "done"]:
        print("Okay Bye")
        break

    if (
        conversation_memory["last_question"]
        and normalize_query(query) == normalize_query(conversation_memory["last_question"])
    ):
        print("User  :  ", query)
        print("Assistant  :  ", conversation_memory["last_answer"])
        print(
            f"Metadata  :\n  Source Type: {conversation_memory['last_source_type']},\n Source: {conversation_memory['last_source']},\n Chunk ID: {conversation_memory['last_chunk_id']},\n Chunk Length: {conversation_memory['last_chunk_length']}"
        )
        print("\n" * 3)
        continue

    initial_state = {
        "query": query,
        "last_question": conversation_memory["last_question"],
        "last_answer": conversation_memory["last_answer"],
    }
    final_state = workflow.invoke(initial_state)

    print("User  :  ", query)
    print("Assistant  :  ", final_state["answeragent"])
    print(
        f"Metadata  :\n Source Type: {final_state['source_type']},\n Source: {final_state['source']},\n Chunk ID: {final_state['chunk_id']},\n Chunk Length: {final_state['chunk_length']}"
    )

    conversation_memory["last_question"] = query
    conversation_memory["last_answer"] = final_state["answeragent"]
    conversation_memory["last_source_type"] = final_state["source_type"]
    conversation_memory["last_source"] = final_state["source"]
    conversation_memory["last_chunk_id"] = final_state["chunk_id"]
    conversation_memory["last_chunk_length"] = final_state["chunk_length"]

    print("\n" * 3)
