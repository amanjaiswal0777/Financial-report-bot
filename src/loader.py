"""Read PDF pages, using OCR when a page has little or no embedded text."""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


_MIN_NATIVE_TEXT_CHARS = 80
_OCR_ZOOM = 2.5


def _ocr_page(page, ocr_engine) -> str:
    """Render a PDF page and return OCR text, without requiring Pillow."""
    import pymupdf
    import numpy as np

    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(_OCR_ZOOM, _OCR_ZOOM), alpha=False)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    result, _ = ocr_engine(image)
    if not result:
        return ""
    return "\n".join(str(line[1]) for line in result if len(line) > 1 and line[1])


def load_pdf(path: Path, use_ocr: bool = True):
    docs = PyPDFLoader(str(path)).load()
    ocr_engine = None
    pages_requiring_ocr = [
        i for i, doc in enumerate(docs)
        if len((doc.page_content or "").strip()) < _MIN_NATIVE_TEXT_CHARS
    ]

    if use_ocr and pages_requiring_ocr:
        try:
            import pymupdf
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "This PDF contains pages without extractable text. Install the OCR "
                "dependencies with `pip install -r requirements.txt`, then rerun ingest.py."
            ) from exc

        pdf = pymupdf.open(str(path))
        ocr_engine = RapidOCR()
        try:
            for i in pages_requiring_ocr:
                print(f"  OCR page {i + 1}/{len(docs)}")
                text = _ocr_page(pdf[i], ocr_engine)
                if text.strip():
                    native_text = (docs[i].page_content or "").strip()
                    docs[i].page_content = "\n\n".join(
                        part for part in (native_text, text) if part
                    )
        finally:
            pdf.close()

    for doc in docs:
        doc.metadata["source"] = path.name
        # PyPDFLoader pages start at 0; make it 1-based (PDF viewer page number)
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
    return docs
