"""ChromaDB persistent client with a single collection."""
from pathlib import Path
import chromadb

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "documents"

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection():
    # get_or_create avoids the need for an explicit init step.
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
