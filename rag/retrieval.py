"""Query-time retrieval against ChromaDB."""
from typing import List, Dict, Any
from rag.embeddings import get_collection
from backend.gemini_service import embed_query


def retrieve(question: str, top_k: int = 4) -> List[Dict[str, Any]]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    qvec = embed_query(question)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks: List[Dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": doc,
            "source": (meta or {}).get("source", "unknown"),
            "chunk_index": (meta or {}).get("chunk_index", -1),
            "score": round(1.0 - float(dist), 4) if dist is not None else None,
        })
    return chunks
