"""Local policy/SOP retrieval using Sentence Transformers and FAISS."""
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import KNOWLEDGE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_K


def load_knowledge_documents():
    docs = []
    for path in sorted(KNOWLEDGE_DIR.glob("*")):
        try:
            if path.suffix.lower() in {".md", ".txt"}:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
            elif path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
        except Exception as exc:
            print(f"Skipping knowledge file {path.name}: {exc}")
    return docs


def retrieve_policy_context(query: str, k: int = RETRIEVAL_K) -> list[str]:
    docs = load_knowledge_documents()
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return []
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = FAISS.from_documents(chunks, embeddings)
    matches = store.similarity_search(query, k=min(k, len(chunks)))
    return [
        f"Source: {Path(doc.metadata.get('source', 'knowledge source')).name}\n{doc.page_content}"
        for doc in matches
    ]
