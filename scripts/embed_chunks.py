import requests
import json
import time
from pathlib import Path

CHUNK_DIR = "/workspace/chunks"
EMBED_FILE = "/workspace/embeddings.jsonl"

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

MAX_CHARS = 1500      # SAFE for nomic
RETRIES = 3
SLEEP_BETWEEN = 1.5


def safe_chunks(text, size=MAX_CHARS):
    chunks = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > 0 else size
        chunks.append(text[:cut])
        text = text[cut:]
    chunks.append(text)
    return chunks


def embed(text):
    payload = {
        "model": MODEL,
        "prompt": text.strip()
    }

    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["embedding"]
        except Exception as e:
            print(f"      ⚠ embedding failed (attempt {attempt}): {e}")
            time.sleep(2 * attempt)

    raise RuntimeError("Embedding failed after retries")


def main():
    out = open(EMBED_FILE, "w", encoding="utf-8")

    for file in Path(CHUNK_DIR).glob("*_chunks.txt"):
        content = file.read_text(encoding="utf-8")
        raw_chunks = content.split("\n\n---CHUNK---\n\n")

        for i, chunk in enumerate(raw_chunks):
            sub_chunks = safe_chunks(chunk)

            for j, sub in enumerate(sub_chunks):
                print(f"Embedding {file.name} chunk {i+1}.{j+1}")
                vec = embed(sub)

                record = {
                    "text": sub,
                    "embedding": vec,
                    "source": file.stem,
                    "chunk_id": f"{i}.{j}"
                }
                out.write(json.dumps(record) + "\n")

                time.sleep(SLEEP_BETWEEN)

    out.close()
    print("Embeddings completed successfully")


if __name__ == "__main__":
    main()
