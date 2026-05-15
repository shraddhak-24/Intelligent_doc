"""FastAPI routes: /upload, /ask, /feedback, /history."""
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import Document
from backend import memory as memory_store
from backend import gemini_service
from rag.ingest import ingest_pdf
from rag.retrieval import retrieve

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    session_id: str
    question: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1 (thumbs up) or -1 (thumbs down)
    correction: Optional[str] = None


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted.")

    file_path = UPLOAD_DIR / file.filename
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    file_path.write_bytes(contents)

    try:
        num_chunks, full_text = ingest_pdf(str(file_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {e}")

    if num_chunks == 0:
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF.")

    doc = Document(filename=file.filename, num_chunks=num_chunks)
    db.add(doc)
    db.commit()

    summary = ""
    try:
        summary = gemini_service.summarize_text(full_text)
    except Exception:
        summary = ""

    return {
        "filename": file.filename,
        "num_chunks": num_chunks,
        "summary": summary,
    }


@router.post("/ask")
def ask(req: AskRequest, db: Session = Depends(get_db)):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    history = memory_store.get_short_term(db, req.session_id)

    try:
        chunks = retrieve(question, top_k=4)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    try:
        result = gemini_service.generate_answer(question, chunks, history)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini generation failed: {e}")

    answer = result["answer"]
    confidence = result["confidence"]

    sources = [
        {
            "filename": c["source"],
            "chunk_index": c["chunk_index"],
            "preview": c["text"][:250] + ("..." if len(c["text"]) > 250 else ""),
            "score": c.get("score"),
        }
        for c in chunks
    ]

    memory_store.save_message(db, req.session_id, "user", question)
    memory_store.save_message(db, req.session_id, "assistant", answer, sources=sources)
    memory_store.save_qa(db, req.session_id, question, answer, confidence=confidence)

    followups: List[str] = []
    try:
        followups = gemini_service.suggest_followups(question, answer)
    except Exception:
        followups = []

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "followups": followups,
    }


@router.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    if req.rating not in (-1, 1):
        raise HTTPException(status_code=400, detail="rating must be 1 or -1.")
    qa = memory_store.record_feedback(db, req.question, req.answer, req.rating, req.correction)
    return {"ok": True, "qa_id": qa.id}


@router.get("/history/{session_id}")
def history(session_id: str, db: Session = Depends(get_db)):
    return {"session_id": session_id, "messages": memory_store.get_episodic(db, session_id)}


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    rows = db.query(Document).order_by(Document.id.desc()).all()
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "num_chunks": r.num_chunks,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        }
        for r in rows
    ]
