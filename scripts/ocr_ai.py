import pdfplumber
import pytesseract
import cv2
import numpy as np
import requests
import json
from pathlib import Path
from PIL import Image

# ---------------- CONFIG ----------------

INPUT_PDF_DIR = "/workspace/input_pdfs"
OUTPUT_DIR = "/workspace/output_final"
PROGRESS_FILE = "/workspace/progress.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "phi4-mini"

MAX_AI_CHARS = 8000   # safe for RTX 4090

# Tesseract config (good for tables + legal text)
TESS_CONFIG = "--oem 3 --psm 4"

# ----------------------------------------


def load_progress():
    if Path(PROGRESS_FILE).exists():
        return json.loads(Path(PROGRESS_FILE).read_text())
    return {}


def save_progress(progress):
    Path(PROGRESS_FILE).write_text(json.dumps(progress, indent=2))


def split_text(text, size=MAX_AI_CHARS):
    chunks = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > 0 else size
        chunks.append(text[:cut])
        text = text[cut:]
    chunks.append(text)
    return chunks


def preprocess_image(pil_img):
    if pil_img is None:
        return None

    img = np.array(pil_img)
    if img.size == 0:
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(2.0, (8, 8))
    img = clahe.apply(img)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(img)


def correct_with_ollama(text):
    chunks = split_text(text)
    corrected = []

    for i, chunk in enumerate(chunks, 1):
        print(f"    AI chunk {i}/{len(chunks)}")

        prompt = f"""
# Role:
You are a highly accurate text editor specializing in OCR-corrected documents.

# Objective:
Produce a clean, corrected version of the provided text while preserving its original meaning and technical accuracy.

# Context:
The input text may contain spelling, grammar, punctuation, and common OCR-related errors caused by character misrecognition during scanning or digitization.

# Instructions:

## Instruction 1:
Correct all spelling, grammar, and punctuation errors in the text.

## Instruction 2:
Fix common OCR mistakes (for example: "rn" → "m", "cl" → "d", incorrect character substitutions, merged or broken words).

## Instruction 3:
Preserve the original meaning, tone, formatting, and all technical or domain-specific terms.

# Notes:
- Return only the corrected text.
- Do not include explanations, comments, or metadata.
- Do not rewrite content beyond necessary corrections.

Text to correct:
{chunk}

Corrected text:
"""

        payload = {
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }

        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        corrected.append(r.json()["response"])

    return "\n".join(corrected)


def process_pdf(pdf_path, progress):
    name = pdf_path.name
    if progress.get(name) == "done":
        print(f"Skipping {name} (already processed)")
        return

    print(f"\nProcessing PDF: {name}")
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            print(f"  Page {i}/{len(pdf.pages)}")

            # Native text (best quality)
            native = page.extract_text()
            if native and native.strip():
                all_text.append(native)

            # OCR images (tables, scans, landscape)
            for img in page.images:
                x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                px0, ptop, px1, pbottom = page.bbox

                x0, top = max(px0, x0), max(ptop, top)
                x1, bottom = min(px1, x1), min(pbottom, bottom)

                if x1 <= x0 or bottom <= top:
                    continue

                try:
                    pil_img = page.crop((x0, top, x1, bottom)).to_image().original
                    proc = preprocess_image(pil_img)
                    if proc:
                        text = pytesseract.image_to_string(proc, config=TESS_CONFIG)
                        if text.strip():
                            all_text.append(text)
                except Exception:
                    continue

    print("  Running AI correction...")
    final_text = correct_with_ollama("\n".join(all_text))

    out_file = Path(OUTPUT_DIR) / f"{pdf_path.stem}_final.txt"
    out_file.write_text(final_text, encoding="utf-8")

    progress[name] = "done"
    save_progress(progress)
    print(f"Saved: {out_file.name}")


def main():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    progress = load_progress()

    for pdf in Path(INPUT_PDF_DIR).glob("*.pdf"):
        process_pdf(pdf, progress)


if __name__ == "__main__":
    main()
