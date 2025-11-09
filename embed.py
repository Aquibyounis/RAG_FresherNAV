import json
import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# CONFIG
# -------------------------------
DATASET_FOLDER = "dataset"
DB_DIR = "new_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize embeddings and Chroma DB
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# -------------------------------
# LOAD JSON FILE NAMES FROM names.txt
# -------------------------------
with open("names.txt", "r", encoding="utf-8") as f:
    json_files = [line.strip() for line in f if line.strip()]

print(f"Found {len(json_files)} JSON files:", json_files)

# -------------------------------
# PROCESS EACH JSON FILE
# -------------------------------
for json_file in json_files:
    path = os.path.join(DATASET_FOLDER, json_file)
    print(f"\n📌 Processing {json_file}...")

    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    texts = []
    metadatas = []

    for entry in entries:

        # ✅ extract text safely
        text_content = entry.get("text", "")
        if not text_content:
            continue
        
        texts.append(text_content)

        # ✅ Fix metadata fields
        meta = {
            "title": str(entry.get("title", "")),
            "category": str(entry.get("category", "")),
            "type": str(entry.get("type", "")),
        }

        # ✅ Convert list tags to string if needed
        tags = entry.get("tags", "")
        if isinstance(tags, list):
            tags = ", ".join([str(t) for t in tags])
        meta["tags"] = tags

        metadatas.append(meta)

    # ✅ Add to DB
    if texts:
        db.add_texts(texts, metadatas=metadatas)
        print(f"✅ Stored {len(texts)} entries")

# -------------------------------
# DATABASE IS AUTOMATICALLY PERSISTED
# -------------------------------
print("\n🎯 Embeddings successfully created & stored in:", DB_DIR)
