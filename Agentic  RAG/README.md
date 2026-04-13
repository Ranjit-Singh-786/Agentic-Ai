# 🤖 RAG Agent — Retrieval-Augmented Generation with LangGraph

A conversational AI assistant that answers questions strictly from a ChromaDB vector knowledge base, built with LangGraph, LangChain, and GPT-4o-mini.

---

## 📌 Overview

This project implements a **RAG (Retrieval-Augmented Generation)** pipeline using a stateful **LangGraph workflow**. The agent retrieves relevant document chunks from a ChromaDB vector store and generates grounded answers — it never invents information outside the knowledge base.

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│           LangGraph Workflow            │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │        Retrieval Agent           │   │──── ChromaDB (398 chunks)
│  │  ReAct Agent + RetrievalTool     │   │
│  └──────────────┬───────────────────┘   │
│                 │                       │
│                 ▼                       │
│  ┌──────────────────────────────────┐   │
│  │         Answer Agent             │   │
│  │  GPT-4o-mini + Prompt Rules      │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │
    ▼
Response + Metadata → Conversation Memory
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Retrieval-Grounded Answers** | All answers are based strictly on ChromaDB context |
| 🔁 **Repeat Question Detection** | LLM detects repeated/similar questions and returns cached answer |
| ❌ **Retrieval Failure Handling** | Stops and returns fallback after 2 consecutive retrieval failures |
| ✅ **Session Resolution Signal** | Adds closing message when query is fully resolved |
| 👋 **Greeting Handling** | Responds warmly to greetings without using retrieved context |
| 🧠 **Conversation Memory** | Maintains last Q&A across turns for continuity |
| 📊 **Metadata Output** | Shows source type, source, chunk ID, and chunk length per response |

---

## 🧠 Logic Rules (in AnswerAgent Prompt)

1. **Greeting** → Warm reply, ask what user wants to know
2. **Repeat / similar question** → Politely acknowledge, return last answer as-is
3. **Answer from retrieved context only** → No outside knowledge
4. **No context found** → Return exact fallback: *"I don't have enough information in the knowledge base to answer."*
5. **No invention** → Never assume or hallucinate
6. **Fully resolved answer** → Append: *"If you need anything else, feel free to ask. Otherwise, I'll end the session."*

---

## 🗂️ Project Structure

```
project/
│
├── rag_agent.py              # Main application file
├── .env                      # API keys (not committed)
├── requirements.txt          # Python dependencies
└── chroma_sbert_db/          # ChromaDB vector store directory
    └── pdf_url_sbert_collection
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-agent.git
cd rag-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Prepare ChromaDB

Make sure your ChromaDB vector store exists at `./chroma_sbert_db` with the collection name `pdf_url_sbert_collection`. If you need to ingest documents, run your ingestion script before starting the agent.

### 6. Run the agent

```bash
python rag_agent.py
```

---

## 📦 Dependencies

```txt
langchain
langchain-community
langchain-openai
langgraph
chromadb
sentence-transformers
python-dotenv
```

Install all at once:

```bash
pip install langchain langchain-community langchain-openai langgraph chromadb sentence-transformers python-dotenv
```

---

## 💬 Usage Example

```
Connected! Total chunks: 398

Enter your query  : What is machine learning?
User       :  What is machine learning?
Assistant  :  Machine learning is a subset of AI that enables systems to learn from data...
             If you need anything else, feel free to ask. Otherwise, I'll end the session.
Metadata   :
  Source Type  : pdf
  Source       : ml_basics.pdf
  Chunk ID     : chunk_042
  Chunk Length : 198


Enter your query  : What is machine learning?
User       :  What is machine learning?
Assistant  :  You already asked this earlier. Here is the same answer:
             Machine learning is a subset of AI...


Enter your query  : bye
Thank You, Nice to meet you!
```

---

## 🔧 Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `CHROMA_DIR` | `rag_agent.py` | `./chroma_sbert_db` | ChromaDB persist directory |
| `COLLECTION_NAME` | `rag_agent.py` | `pdf_url_sbert_collection` | Collection to query |
| `model_name` | `rag_agent.py` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `k` | `RetrievalTool` | `2` | Number of chunks to retrieve |
| `model` | `ChatOpenAI` | `gpt-4o-mini` | LLM for answer generation |

---

## 🛑 Exit Keywords

Type any of the following to end the session gracefully:

```
thanks  /  good  /  bye  /  done  /  okay
```

---

## 📝 Notes

- The embedding model runs on **CPU** by default. Change `device` to `"cuda"` for GPU acceleration.
- The agent automatically ends the session after **2 consecutive retrieval failures**.
- All answers are strictly grounded — the agent will not answer from general knowledge.

---

## 📄 License

MIT License — feel free to use and modify.
