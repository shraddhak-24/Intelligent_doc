"""Thin wrapper around Google Gemini for generation + embedding."""
import os
from typing import List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GENERATION_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"

if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)


def _ensure_configured():
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to your .env file."
        )


def embed_texts(texts: List[str], task_type: str = "retrieval_document") -> List[List[float]]:
    """Embed a list of strings. task_type is 'retrieval_document' or 'retrieval_query'."""
    _ensure_configured()
    vectors: List[List[float]] = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        vectors.append(result["embedding"])
    return vectors


def embed_query(text: str) -> List[float]:
    return embed_texts([text], task_type="retrieval_query")[0]


SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question using ONLY the provided context chunks and the recent conversation history.

Rules:
- If the answer is not in the context, say you don't know based on the documents.
- Be concise and accurate.
- When relevant, reference the source filename in your answer.
- Use the conversation history to resolve follow-up questions like "explain more" or "what about that".

After your answer, on a new line output exactly: CONFIDENCE: <a number 0-100 representing how well the context supports your answer>
"""


def generate_answer(question: str, context_chunks: list, history: list) -> dict:
    """Call Gemini with retrieved context + short-term history. Returns {answer, confidence}."""
    _ensure_configured()
    model = genai.GenerativeModel(GENERATION_MODEL, system_instruction=SYSTEM_PROMPT)

    context_block = "\n\n".join(
        f"[Source: {c['source']} | chunk #{c['chunk_index']}]\n{c['text']}"
        for c in context_chunks
    ) or "(no documents retrieved)"

    history_block = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    ) or "(no prior turns)"

    prompt = (
        f"CONVERSATION HISTORY:\n{history_block}\n\n"
        f"CONTEXT CHUNKS:\n{context_block}\n\n"
        f"USER QUESTION: {question}"
    )

    try:
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
    except Exception as e:
        return {"answer": f"Gemini error: {e}", "confidence": 0.0}

    confidence = 0.0
    answer = raw
    if "CONFIDENCE:" in raw:
        answer, _, conf_part = raw.rpartition("CONFIDENCE:")
        answer = answer.strip()
        try:
            confidence = float("".join(ch for ch in conf_part if ch.isdigit() or ch == "."))
        except ValueError:
            confidence = 0.0

    return {"answer": answer, "confidence": confidence}


def summarize_text(text: str) -> str:
    """Bonus: short summary of a document."""
    _ensure_configured()
    model = genai.GenerativeModel(GENERATION_MODEL)
    truncated = text[:12000]
    resp = model.generate_content(
        f"Summarize the following document in 5-7 bullet points:\n\n{truncated}"
    )
    return (resp.text or "").strip()


def suggest_followups(question: str, answer: str) -> List[str]:
    """Bonus: produce 3 follow-up question ideas."""
    _ensure_configured()
    model = genai.GenerativeModel(GENERATION_MODEL)
    prompt = (
        "Given the question and answer below, suggest exactly 3 short follow-up questions "
        "the user might ask next. Return them as a plain numbered list, no extra commentary.\n\n"
        f"Q: {question}\nA: {answer}"
    )
    try:
        resp = model.generate_content(prompt)
        lines = [ln.strip(" -0123456789.") for ln in (resp.text or "").splitlines() if ln.strip()]
        return [ln for ln in lines if ln][:3]
    except Exception:
        return []
