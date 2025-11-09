# main2.py — Mistral RAG ENGINE (no FastAPI)

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
MODEL_NAME = "mistral"
TOP_K = 3

chat_history = []


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_db():
    from langchain_chroma import Chroma
    return Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())


def build_prompt(context: str, question: str) -> str:
    history_text = "\n".join(
        [f"User: {u}\nCampusGuide: {a}" for u, a in chat_history]
    )

    template = f"""
SYSTEM:
You are CampusGuide. Be clear, direct and factual.
Answer in 2–3 crisp sentences. If unsure, say so and recommend official sources.

CONTEXT:
{context}

CHAT HISTORY:
{history_text}

USER QUESTION:
{question}

ANSWER:
"""
    return template


def get_filter(question: str):
    q = question.lower()
    if "leave" in q or "outing" in q or "weekend" in q:
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


def model_invoke_mistral(prompt: str) -> str:
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


def run_query(question: str) -> str:
    q = question.strip()
    if not q:
        return ""

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

    if not context:
        context = ""

    prompt = build_prompt(context, q)
    answer = model_invoke_mistral(prompt)

    answer = answer.replace("ASSISTANT:", "").strip()

    chat_history.append((q, answer))

    return answer
