# Legal RAG System (Offline, OCR-based)

This project implements a fully offline **Retrieval-Augmented Generation (RAG)** system for Indian legal Acts, Rules, and Regulations using open-source models only.

The system was designed to avoid API rate limits, data leakage, and cloud dependency issues by using **local LLMs via Ollama**, **FAISS vector search**, and **OCR-based text extraction**.

---

## Problem Statement

Government legal PDFs often:
- Contain scanned images instead of text
- Mix portrait and landscape pages
- Include complex tables and section formatting
- Produce OCR spelling errors

This project builds a robust pipeline to:
1. Extract text accurately
2. Correct OCR errors using LLMs
3. Chunk legal text semantically
4. Store embeddings locally
5. Answer legal questions strictly from source Acts

---

## End-to-End Pipeline

### 1. PDF Text Extraction (OCR + Native Text)
- Used `pdfplumber`, `pytesseract`, and OpenCV
- Handles:
  - Image-only PDFs
  - Mixed image + text pages
  - Tables
  - Portrait and landscape layouts

**Script:** `scripts/ocr_ai.py`

---

### 2. OCR Error Correction using LLM (Offline)
- OCR output may contain errors like:
  - `rn → m`
  - `cl → d`
  - broken words
- Corrected using **open-source LLMs via Ollama**
- Avoided OpenAI/Gemini APIs due to rate limits

**Script:** `scripts/ocr_ai.py`  
**Models used:** `phi4-mini`, `mistral`, fallback local models

---

### 3. Recursive + Semantic Chunking
- Legal text chunked by:
  - Section boundaries
  - Sub-sections and clauses
  - Token limits
- Preserves legal structure for accurate retrieval

**Script:** `scripts/chunk_text.py`

---

### 4. Embedding Generation (Local)
- Used **nomic-embed-text** via Ollama
- Fully offline embedding generation
- One embedding per legal chunk

**Script:** `scripts/embed_chunks.py`

---

### 5. Vector Index Creation
- Stored embeddings in **FAISS**
- Metadata stored separately (chunk text + source)

**Script:** `scripts/build_faiss.py`  
**Outputs (ignored in Git):**
- `legal_index.faiss`
- `legal_meta.json`

---

### 6. Legal Question Answering (RAG)
- User question → embedded → FAISS similarity search
- Retrieved legal chunks passed to LLM
- LLM instructed to:
  - Quote **verbatim**
  - Use **only retrieved text**
  - Say *“Not found in provided documents”* if missing

**Script:** `scripts/legal_rag.py`

---

## Technologies Used

- Python 3.10+
- Tesseract OCR
- OpenCV
- FAISS
- Ollama (local LLM runner)
- nomic-embed-text
- Open-source LLMs (phi4-mini, mistral, etc.)

---

## Why Offline & Open-Source?

- Avoid API rate limits
- Avoid legal data leakage
- Full control over embeddings and inference
- Reproducible pipeline
- Cost-effective for large legal corpora

---

## Project Status

- OCR extraction: ✅ Complete
- Text correction: ✅ Complete
- Chunking: ✅ Complete
- Vector index: ✅ Built
- RAG querying: 🧪 Testing / tuning

---

## Disclaimer

This system provides **text retrieval only** from legal documents.  
It does **not** provide legal advice or interpretation.
