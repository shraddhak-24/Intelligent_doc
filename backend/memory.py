"""Memory layer: short-term, long-term, and episodic memory backed by SQLite."""
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from backend.models import ChatMessage, QAMemory

SHORT_TERM_LIMIT = 5  # last N turns (user+assistant pairs combined)


def save_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
) -> ChatMessage:
    msg = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        sources=json.dumps(sources) if sources else None,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_short_term(db: Session, session_id: str, limit: int = SHORT_TERM_LIMIT) -> List[Dict[str, str]]:
    """Last N messages, oldest first, ready to feed to the LLM."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    rows = list(reversed(rows))
    return [{"role": r.role, "content": r.content} for r in rows]


def get_episodic(db: Session, session_id: str) -> List[Dict[str, Any]]:
    """Full ordered chat history for the session."""
    rows = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .all()
    )
    out = []
    for r in rows:
        out.append({
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "sources": json.loads(r.sources) if r.sources else [],
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })
    return out


def save_qa(
    db: Session,
    session_id: str,
    question: str,
    answer: str,
    confidence: Optional[float] = None,
) -> QAMemory:
    qa = QAMemory(
        session_id=session_id,
        question=question,
        answer=answer,
        confidence=confidence,
    )
    db.add(qa)
    db.commit()
    db.refresh(qa)
    return qa


def record_feedback(
    db: Session,
    question: str,
    answer: str,
    rating: int,
    correction: Optional[str] = None,
) -> QAMemory:
    """Attach feedback to the most recent matching QA row, or create one if none exists."""
    qa = (
        db.query(QAMemory)
        .filter(QAMemory.question == question, QAMemory.answer == answer)
        .order_by(QAMemory.id.desc())
        .first()
    )
    if qa is None:
        qa = QAMemory(session_id="orphan", question=question, answer=answer)
        db.add(qa)
    qa.feedback_score = rating
    if correction:
        qa.correction = correction
    db.commit()
    db.refresh(qa)
    return qa
