import json
import faiss
import numpy as np
from pathlib import Path

EMBED_FILE = "/workspace/embeddings.jsonl"
FAISS_INDEX = "/workspace/legal_index.faiss"
META_FILE = "/workspace/legal_meta.json"


def main():
    vectors = []
    metadata = []

    for line in Path(EMBED_FILE).read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        vectors.append(obj["embedding"])
        metadata.append({
            "text": obj["text"],
            "source": obj["source"],
            "chunk_id": obj["chunk_id"]
        })

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, FAISS_INDEX)
    Path(META_FILE).write_text(json.dumps(metadata, indent=2))

    print(f"FAISS index built with {len(vectors)} vectors")


if __name__ == "__main__":
    main()
