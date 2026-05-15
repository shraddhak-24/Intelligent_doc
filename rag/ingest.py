"""PDF -> text -> chunks -> embeddings -> Chroma."""
import uuid
from typing import Tuple
from pypdf import PdfReader

from rag.chunking import chunk_text
from rag.embeddings import get_collection
from backend.gemini_service import embed_texts


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()


def ingest_pdf(pdf_path: str) -> Tuple[int, str]:
    """Read a PDF, embed its chunks, store them. Returns (num_chunks, full_text)."""
    full_text = extract_text(pdf_path)
    if not full_text:
        return 0, ""

    chunks = chunk_text(full_text)
    if not chunks:
        return 0, full_text

    filename = pdf_path.replace("\\", "/").split("/")[-1]
    vectors = embed_texts(chunks, task_type="retrieval_document")

    collection = get_collection()
    ids = [f"{filename}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": filename, "chunk_index": i} for i in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks), full_text
