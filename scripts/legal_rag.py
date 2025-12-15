import faiss
import json
import numpy as np
import requests

# ---------------- CONFIG ----------------
BASE_DIR = "C:/legal_rag"   # change if needed

FAISS_INDEX = f"{BASE_DIR}/legal_index.faiss"
META_FILE = f"{BASE_DIR}/legal_meta.json"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "mistral"   # recommended for CPU (can use llama3.1:8b)
TOP_K = 3                # keep low for speed + precision
# ---------------------------------------


def embed_query(text: str) -> np.ndarray:
    r = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        },
        timeout=60
    )
    r.raise_for_status()
    return np.array(r.json()["embedding"], dtype="float32")


def ask_llm(context: str, question: str) -> str:
    prompt = f"""
Role:
You are a Legal Question-Answering Assistant operating within a Retrieval-Augmented Generation (RAG) system over statutory legal texts stored in a vector database.

Objective:
Answer user questions strictly and exclusively based on the provided legal text retrieved from the vector database, without introducing any external knowledge, interpretation, or assumptions.

Context:
The system contains converted legal Acts, Rules, and Regulations stored as embeddings.
Each query retrieves relevant statutory provisions which must be treated as the sole source of truth for generating responses.

Instructions:

Instruction 1:
Answer the question only using the legal text provided below.
Do not use general legal knowledge, prior training data, or personal reasoning.

Instruction 2:
If the answer is not explicitly present in the provided legal text, respond with exactly:
"Not found in provided documents"

Instruction 3:
Respond only by quoting verbatim from the Act, including relevant Sections, Sub-sections, Clauses, Provisos, or Explanations.
Do not paraphrase, summarize, or add explanations.

Notes:
Use the original numbering, punctuation, capitalization, and wording exactly as it appears in the Act.
If multiple provisions are relevant, quote each provision separately and verbatim.

----------------------------------------
LEGAL TEXT (SOLE SOURCE):
{context}

----------------------------------------
QUESTION:
{question}

ANSWER (verbatim only):
"""

    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "options": {
                "temperature": 0.0,
                "num_ctx": 2048
            }
        },
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def main():
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(FAISS_INDEX)
    meta = json.load(open(META_FILE, encoding="utf-8"))

    print("Legal RAG ready. Type 'exit' to quit.")

    while True:
        question = input("\nAsk a legal question: ").strip()
        if question.lower() == "exit":
            break

        query_vec = embed_query(question)
        _, ids = index.search(query_vec.reshape(1, -1), TOP_K)

        retrieved_chunks = [meta[i]["text"] for i in ids[0]]
        context = "\n\n".join(retrieved_chunks)

        answer = ask_llm(context, question)
        print("\nANSWER:\n")
        print(answer)


if __name__ == "__main__":
    main()
