import re
from pathlib import Path

INPUT_DIR = "/workspace/output_final"
CHUNK_DIR = "/workspace/chunks"
Path(CHUNK_DIR).mkdir(exist_ok=True)

MAX_CHARS = 1500   # good for embeddings
MIN_CHARS = 500


def semantic_split(text):
    patterns = [
        r"\n\s*CHAPTER\s+[IVXLC]+\b",
        r"\n\s*Chapter\s+[IVXLC]+\b",
        r"\n\s*SECTION\s+\d+[A-Z]?",
        r"\n\s*Section\s+\d+[A-Z]?",
        r"\n\s*\d+\.\s+"   # numbered clauses
    ]
    regex = "|".join(patterns)
    parts = re.split(regex, text)
    return [p.strip() for p in parts if p.strip()]


def paragraph_split(text):
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


def recursive_chunk(units):
    chunks = []
    buffer = ""

    for u in units:
        if len(buffer) + len(u) <= MAX_CHARS:
            buffer += "\n\n" + u
        else:
            if len(buffer) >= MIN_CHARS:
                chunks.append(buffer.strip())
                buffer = u
            else:
                buffer += "\n\n" + u

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


def process_file(txt_file):
    text = txt_file.read_text(encoding="utf-8").strip()

    # Step 1: semantic split
    semantic_parts = semantic_split(text)
    chunks = recursive_chunk(semantic_parts)

    # Step 2: fallback if chunking failed
    if len(chunks) <= 1:
        paras = paragraph_split(text)
        chunks = recursive_chunk(paras)

    out_file = Path(CHUNK_DIR) / f"{txt_file.stem}_chunks.txt"
    out_file.write_text("\n\n---CHUNK---\n\n".join(chunks), encoding="utf-8")

    print(f"Chunked: {txt_file.name} → {len(chunks)} chunks")


def main():
    for txt in Path(INPUT_DIR).glob("*_final.txt"):
        process_file(txt)


if __name__ == "__main__":
    main()
