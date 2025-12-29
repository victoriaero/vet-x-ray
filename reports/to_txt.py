import os
from pathlib import Path
from typing import Optional

DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "text"
GLOB_PATTERN = "*.pdf"

def extract_text_pdfminer(pdf_path: Path) -> str:
    from pdfminer.high_level import extract_text
    try:
        return extract_text(str(pdf_path)) or ""
    except Exception as e:
        print(f"[pdfminer] erro em '{pdf_path.name}': {e}")
        return ""

def extract_text_ocr(pdf_path: Path) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        return ""

    text_parts = []
    try:
        images = convert_from_path(str(pdf_path))
        for img in images:
            text_parts.append(pytesseract.image_to_string(img))
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"[OCR] erro em '{pdf_path.name}': {e}")
        return ""

def convert_one(pdf_path: Path, out_path: Path, use_ocr_if_empty: bool = True) -> bool:
    text = extract_text_pdfminer(pdf_path).strip()

    if not text and use_ocr_if_empty:
        ocr_text = extract_text_ocr(pdf_path).strip()
        if ocr_text:
            text = ocr_text

    try:
        out_path.write_text(text, encoding="utf-8")
        return True
    except Exception as e:
        print(f"[write] erro escrevendo '{out_path.name}': {e}")
        return False

def main():
    if not DATA_DIR.exists():
        print(f"Pasta '{DATA_DIR}' não encontrada.")
        return

    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(DATA_DIR.glob(GLOB_PATTERN))
    pdfs = [p for p in pdfs if p.parent != TEXT_DIR]

    if not pdfs:
        print(f"Nenhum PDF encontrado em '{DATA_DIR}'.")
        return

    converted, skipped = 0, 0
    for pdf in pdfs:
        out_txt = TEXT_DIR / (pdf.stem + ".txt")

        if out_txt.exists() and out_txt.stat().st_mtime > pdf.stat().st_mtime:
            skipped += 1
            continue

        ok = convert_one(pdf, out_txt, use_ocr_if_empty=True)
        converted += int(ok)

        status = "ok" if ok else "falhou"
        print(f"[{status}] {pdf.name} -> {out_txt.relative_to(DATA_DIR.parent)}")

    print(f"\nConvertidos: {converted} | Pulados: {skipped} | Total PDFs: {len(pdfs)}")
    print(f"Arquivos .txt em: {TEXT_DIR}")

if __name__ == "__main__":
    main()
