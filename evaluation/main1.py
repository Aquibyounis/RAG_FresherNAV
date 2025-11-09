# main1.py — LLaMA 3.2 RAG ENGINE (no FastAPI)

import os
import re
import subprocess
from functools import lru_cache
from typing import Dict, Optional

# -------------------------------
# CONFIG
# -------------------------------
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "llama3.2"
TOP_K = 3

# -------------------------------
# MEMORY BUFFER
# -------------------------------
chat_history = []  # [("user", msg), ("ai", msg)]


# -------------------------------
# LOAD EMBEDDINGS (lazy)
# -------------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


# -------------------------------
# LOAD CHROMA DB (lazy)
# -------------------------------
@lru_cache(maxsize=1)
def get_db():
    from langchain_chroma import Chroma
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


# -------------------------------
# PROMPT BUILDER
# -------------------------------
def build_prompt(context: str, question: str) -> str:
    history_text = "\n".join(
        [f"User: {u}\nCampusGuide: {a}" for u, a in chat_history]
    )

    template = f"""
SYSTEM:
You are "CampusGuide," a friendly AI assistant for VIT-AP students.
Answer in simple, short sentences (2–3 lines max) with 1–2 emojis.
Use CONTEXT only if relevant. If no context exists, say you are not sure and suggest checking official sources.

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION:
{question}

ANSWER:
"""
    return template


# -------------------------------
# QUESTION FILTERING
# -------------------------------
def get_filter(question: str):
    q = question.lower()
    if re.search(r"\bleave\b|\bouting\b|\bweekend\b", q):
        return {"category": "VTOP"}
    if "placement" in q or "pat" in q:
        return {"category": "Placements"}
    if "hostel" in q:
        return {"category": "Hostels"}
    if "guest" in q:
        return {"category": "GuestHouse"}
    if "startup" in q:
        return {"category": "Startups"}
    return None


# -------------------------------
# MODEL INVOKE (CLI only, no API)
# -------------------------------
def model_invoke_llama(prompt: str) -> str:
    try:
        res = subprocess.run(
            ["ollama", "run", MODEL_NAME],
            input=prompt.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=180
        )
        return res.stdout.decode().strip()
    except:
        return "Model error."


# -------------------------------
# MAIN RAG QUERY
# -------------------------------
def run_query(question: str) -> str:
    q = question.strip()
    if not q:
        return ""

    # ---------------------------
    # STEP 1 → RETRIEVAL
    # ---------------------------
    db = get_db()
    filt = get_filter(q)

    try:
        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": 25, "filter": filt} if filt else {"k": TOP_K, "fetch_k": 25}
        )
        docs = retriever.invoke(q)
    except:
        docs = []

    context = "\n\n".join([d.page_content for d in docs[:TOP_K]]) if docs else ""

    # ---------------------------
    # STEP 2 → FALLBACK IF NO CONTEXT
    # ---------------------------
    if not context:
        context = ""

    # ---------------------------
    # STEP 3 → BUILD PROMPT
    # ---------------------------
    prompt = build_prompt(context, q)

    # ---------------------------
    # STEP 4 → MODEL CALL
    # ---------------------------
    answer = model_invoke_llama(prompt)

    # ---------------------------
    # STEP 5 → CLEAN OUTPUT
    # ---------------------------
    answer = answer.replace("ASSISTANT:", "").strip()

    # ---------------------------
    # STEP 6 → UPDATE MEMORY
    # ---------------------------
    chat_history.append((q, answer))

    return answer
