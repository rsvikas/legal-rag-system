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

---

## Architecture Overview

PDFs
 ↓
OCR + Native Text Extraction
 ↓
LLM-based OCR Correction (Ollama)
 ↓
Recursive + Semantic Chunking
 ↓
Local Embeddings (nomic)
 ↓
FAISS Vector Index
 ↓
RAG Query Pipeline
 ↓
Local LLM Answer (Verbatim Legal Text)

---


---

## Pipeline Details

### 1. OCR and Text Extraction
- Uses `pdfplumber`, `pytesseract`, OpenCV, and PIL
- Handles:
  - Image-only PDFs
  - Mixed image + text pages
  - Tables and structured layouts
  - Portrait and landscape orientations

**Script:** `scripts/ocr_ai.py`

---

### 2. LLM-based OCR Error Correction (Offline)
OCR output may contain errors such as:
- `rn` → `m`
- `cl` → `d`
- Broken or merged words

These are corrected using **open-source LLMs via Ollama**, avoiding paid APIs due to rate limits and reliability issues.

**Script:** `scripts/ocr_ai.py`  
**Models used:** `phi4-mini`, `mistral` (local via Ollama)

---

### 3. Recursive + Semantic Chunking
Legal text is chunked using:
- Section and subsection boundaries
- Clause-aware splitting
- Token-length constraints

This preserves legal structure for accurate retrieval.

**Script:** `scripts/chunk_text.py`

---

### 4. Local Embedding Generation
- Embeddings generated **offline**
- Uses `nomic-embed-text` via Ollama
- One embedding per legal chunk

**Script:** `scripts/embed_chunks.py`

---

### 5. FAISS Vector Index Creation
- Embeddings are stored in a local FAISS index
- Metadata (chunk text + source info) stored separately
- No external vector database is used

**Script:** `scripts/build_faiss.py`

**Generated outputs (intentionally NOT committed to Git):**
- `legal_index.faiss`
- `legal_meta.json`

---

### 6. Legal Question Answering (RAG)
- User question is embedded
- FAISS similarity search retrieves relevant chunks
- Retrieved text is passed to a local LLM with a **strict legal prompt**

The LLM is instructed to:
- Use **only the retrieved legal text**
- Quote **verbatim** from the Act
- Respond with **“Not found in provided documents”** if the answer is missing
- Avoid paraphrasing or legal interpretation

**Script:** `scripts/legal_rag.py`

---

## RAG Prompt Philosophy (Safety-Critical)

The system enforces:
- No hallucinations
- No external legal knowledge
- No summarization or rewording
- Exact quoting with original numbering and punctuation

This makes the system suitable for **legal text retrieval**, not legal advice.

---

## Running the Full Pipeline (Reproducibility)

The scripts must be executed in the following order **on a fresh system**.

### Step 1: OCR and Text Extraction
python scripts/ocr_ai.py
### Step 2: Chunk Legal Text
python scripts/chunk_text.py
### Step 3: Generate Embeddings (Offline)
python scripts/embed_chunks.py
### Step 4: Build FAISS Vector Index
python scripts/build_faiss.py
### Step 5: Query the Legal RAG System
python scripts/legal_rag.py

### Note: Generated files are ignored via .gitignore and must be regenerated locally.

### Running on RunPod (GPU)
Recommended for faster OCR correction and embedding generation.

### Environment
 - PyTorch 2.2
 - CUDA 12+
 - GPU (RTX 4090 recommended)

### Setup
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull phi4-mini
ollama pull nomic-embed-text
pip install -r requirements.txt
```

### Technologies Used
 - Python 3.10+
 - Tesseract OCR
 - OpenCV
 - pdfplumber
 - FAISS
 - Ollama (local LLM runtime)
 - nomic-embed-text
 - Open-source LLMs only

### Why Offline & Open-Source?
 -Avoids API rate limits and failures
 - Prevents legal data leakage
 - Full control over embeddings and inference
 - Cost-effective for large legal corpora
 - Fully reproducible pipeline

### Project Status
 - OCR extraction: ✅ Complete
 - OCR correction: ✅ Complete
 - Semantic chunking: ✅ Complete
 - Embeddings: ✅ Generated
 - FAISS index: ✅ Built
 - RAG querying: 🧪 Testing and tuning

### Limitations
 - OCR accuracy depends on scan quality
 - Tables may lose some visual structure
 - The system retrieves text but does not interpret law
 - No legal advice is provided
   
### Disclaimer
This project is intended for legal text retrieval and research assistance only.
It does not provide legal advice, opinions, or interpretations.
Users must verify all information against official legal sources.




