"""Streamlit UI for the Intelligent Document Q&A System.
Run with:  streamlit run frontend/app.py
"""
import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"
TIMEOUT = 120

st.set_page_config(page_title="Intelligent Document Q&A", page_icon="📄", layout="wide")

# ---------- session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources, confidence, followups}
if "uploads" not in st.session_state:
    st.session_state.uploads = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------- styling ----------
st.markdown(
    """
    <style>
    .stChatMessage { padding: 0.4rem 0.6rem; }
    .source-chip {
        display: inline-block; padding: 2px 8px; margin: 2px 4px 2px 0;
        background: #eef2ff; color: #3730a3; border-radius: 999px; font-size: 0.78rem;
    }
    .confidence { font-size: 0.8rem; color: #6b7280; margin-top: 4px; }
    .followup-btn button { width: 100%; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- sidebar: upload + session info ----------
with st.sidebar:
    st.title("📄 Document Q&A")
    st.caption("RAG + Memory + Gemini")
    st.divider()

    st.subheader("Upload a PDF")
    uploaded = st.file_uploader("Drop a PDF here", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None and st.button("Ingest document", use_container_width=True):
        with st.spinner("Extracting and embedding..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                r = requests.post(f"{API_BASE}/upload", files=files, timeout=TIMEOUT)
                if r.ok:
                    data = r.json()
                    st.success(f"Ingested {data['filename']} — {data['num_chunks']} chunks")
                    st.session_state.uploads.append(data)
                    if data.get("summary"):
                        with st.expander("Summary"):
                            st.write(data["summary"])
                else:
                    st.error(f"Upload failed: {r.json().get('detail', r.text)}")
            except requests.exceptions.RequestException as e:
                st.error(f"Backend unreachable: {e}")

    st.divider()
    st.subheader("Session")
    st.code(st.session_state.session_id, language="text")
    if st.button("New session", use_container_width=True):
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.rerun()

    if st.session_state.uploads:
        st.divider()
        st.subheader("Uploaded this session")
        for u in st.session_state.uploads:
            st.markdown(f"- **{u['filename']}** ({u['num_chunks']} chunks)")

    st.divider()
    if st.button("View full history", use_container_width=True):
        try:
            r = requests.get(
                f"{API_BASE}/history/{st.session_state.session_id}", timeout=TIMEOUT
            )
            if r.ok:
                with st.expander("Episodic memory (full history)", expanded=True):
                    for m in r.json().get("messages", []):
                        st.markdown(f"**{m['role']}** — {m['content']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Backend unreachable: {e}")

# ---------- main chat ----------
st.title("Ask your documents")
st.caption("Short-term memory keeps the last 5 turns. Long-term memory stores Q/A with feedback.")


def _send_feedback(msg, rating, correction=None):
    try:
        question = ""
        for m in reversed(st.session_state.messages):
            if m is msg:
                continue
            if m["role"] == "user":
                question = m["content"]
                break
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "question": question,
                "answer": msg["content"],
                "rating": rating,
                "correction": correction,
            },
            timeout=TIMEOUT,
        )
        st.toast("Feedback recorded", icon="✅")
    except requests.exceptions.RequestException as e:
        st.error(f"Feedback failed: {e}")


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("confidence"):
                st.markdown(
                    f"<div class='confidence'>Confidence: {msg['confidence']:.0f}/100</div>",
                    unsafe_allow_html=True,
                )
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"<span class='source-chip'>{s['filename']} · chunk #{s['chunk_index']}"
                            + (f" · score {s['score']}" if s.get("score") is not None else "")
                            + "</span>",
                            unsafe_allow_html=True,
                        )
                        st.caption(s["preview"])

            # feedback row
            fb_key = f"fb_{idx}"
            cols = st.columns([1, 1, 8])
            with cols[0]:
                if st.button("👍", key=f"up_{idx}"):
                    _send_feedback(msg, 1)
            with cols[1]:
                if st.button("👎", key=f"down_{idx}"):
                    st.session_state[f"{fb_key}_show"] = True

            if st.session_state.get(f"{fb_key}_show"):
                correction = st.text_area(
                    "What should the answer have been? (optional)", key=f"{fb_key}_text"
                )
                if st.button("Submit correction", key=f"{fb_key}_submit"):
                    _send_feedback(msg, -1, correction)
                    st.session_state[f"{fb_key}_show"] = False
                    st.success("Thanks for the feedback.")

            if msg.get("followups"):
                st.markdown("**Suggested follow-ups:**")
                for j, fq in enumerate(msg["followups"]):
                    if st.button(fq, key=f"fu_{idx}_{j}", use_container_width=True):
                        st.session_state.pending_question = fq
                        st.rerun()


# input box (or pending follow-up)
prompt = st.chat_input("Ask a question about your uploaded documents")
if st.session_state.pending_question and not prompt:
    prompt = st.session_state.pending_question
    st.session_state.pending_question = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_BASE}/ask",
                    json={"session_id": st.session_state.session_id, "question": prompt},
                    timeout=TIMEOUT,
                )
                if r.ok:
                    data = r.json()
                    answer = data["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": data.get("sources", []),
                        "confidence": data.get("confidence", 0.0),
                        "followups": data.get("followups", []),
                    })
                    st.rerun()
                else:
                    err = r.json().get("detail", r.text)
                    st.error(f"Error: {err}")
            except requests.exceptions.RequestException as e:
                st.error(f"Backend unreachable: {e}")
