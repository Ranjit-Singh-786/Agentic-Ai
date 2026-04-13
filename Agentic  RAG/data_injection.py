import os
import re
import string
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import warnings 
warnings.filterwarnings("ignore")

# USER AGENT (IMPORTANT)
os.environ["USER_AGENT"] = "Mozilla/5.0"

# CONFIG
PDF_PATH = r"C:\Users\Ranjit\Desktop\Agentic_RAG\Agentic_RAG\MCP_pdf.pdf"
URLS = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
]

CHROMA_DIR = "./chroma_sbert_db"
COLLECTION_NAME = "pdf_url_sbert_collection"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# CLEAN FUNCTION
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

# LOAD PDF
pdf_docs = []
if os.path.exists(PDF_PATH):
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    for doc in pages:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({
            "source_type": "pdf",
            "source": PDF_PATH,
            "file": os.path.basename(PDF_PATH)
        })

    pdf_docs.extend(pages)

# LOAD URL
url_docs = []
if URLS:
    loader = WebBaseLoader(URLS)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update({
            "source_type": "url",
            "source": doc.metadata.get("source", "unknown_url"),
            "file": "web_content"
        })

    url_docs.extend(docs)

# COMBINE
all_docs = pdf_docs + url_docs

# CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(all_docs)

# METADATA
for idx, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = f"chunk_{idx}"
    chunk.metadata["chunk_length"] = len(chunk.page_content)

# EMBEDDING
embedding_model = HuggingFaceEmbeddings(
    model_name= "BAAI/bge-small-en-v1.5" ,        #"sentence-transformers/all-MiniLM-L6-v2",

    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

# STORE
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
    collection_name=COLLECTION_NAME,
)

print(f" Ingestion Complete! Total chunks: {len(chunks)}")
