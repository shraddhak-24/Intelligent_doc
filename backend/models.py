"""SQLAlchemy models for memory + feedback."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from backend.database import Base


class ChatMessage(Base):
    """Episodic memory: every user/assistant turn for a session."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)  # JSON-encoded list of source dicts
    created_at = Column(DateTime, default=datetime.utcnow)


class QAMemory(Base):
    """Long-term memory: distilled Q/A pairs with feedback signal."""
    __tablename__ = "qa_memory"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    feedback_score = Column(Integer, default=0)  # -1, 0, 1
    correction = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Document(Base):
    """Uploaded document registry."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    num_chunks = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
