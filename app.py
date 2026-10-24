"""Streamlit UI for Financial-report-bot.

Run from the project root:
    streamlit run app.py

It reuses your existing code and changes nothing in it:
    ingest  -> src.loader.load_pdf, src.chunker.split_documents, src.vectorstore.build_vectorstore
    ask     -> src.pipeline.answer
"""
import json
import sys
from pathlib import Path


import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (also loads .env)
from src.chunker import split_documents  # noqa: E402
from src.loader import load_pdf  # noqa: E402
from src.pipeline import answer  # noqa: E402
from src.vectorstore import build_vectorstore, load_vectorstore  # noqa: E402

st.set_page_config(page_title="Financial report Q&A", page_icon="📄", layout="wide")

GOLDEN_PATH = ROOT / "eval" / "golden_dataset.json"
NOT_FOUND = "I couldn't find this in the retrieved document text."


# ----------------------------------------------------------------- helpers
def indexed_source():
    """Name of the PDF currently in the Chroma index, or None if there is no index."""
    if not config.CHROMA_DIR.exists():
        return None
    try:
        got = load_vectorstore().get(limit=1, include=["metadatas"])
        metas = got.get("metadatas") or []
        return metas[0].get("source") if metas else None
    except Exception:
        return None


def run_ingest(pdf_path: Path, status, use_ocr: bool = False):
    """Same steps as ingest.py, with progress shown in the UI."""
    # Chroma keeps one client per folder alive in this process. Clear it so the
    # old index can be deleted and rebuilt without "stale database" errors.
    try:
        from chromadb.api.client import SharedSystemClient  # pyright: ignore[reportMissingImports]

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass

    status.update(
        label=(
            "Reading pages and using OCR on scanned pages..."
            if use_ocr
            else "Reading pages without OCR (faster; scanned pages may be skipped)..."
        )
    )
    pages = load_pdf(pdf_path, use_ocr=use_ocr)

    status.update(label=f"Splitting {len(pages)} pages into chunks…")
    chunks = split_documents(pages)

    status.update(label=f"Embedding and indexing {len(chunks)} chunks…")
    build_vectorstore(chunks)
    return len(pages), len(chunks)


def load_sample_questions(n: int = 5):
    if not GOLDEN_PATH.exists():
        return []
    try:
        items = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    picked = [i for i in items if (i.get("additional_metadata") or {}).get("id") in (5, 6, 8, 11, 15)]
    return [i.get("input") or i.get("question") for i in (picked or items)[:n]]


def render_sources(chunks):
    if not chunks:
        return
    pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
    st.caption("Pages retrieved: " + ", ".join(str(p) for p in pages))
    score_label = "rerank score" if config.USE_RERANKER else "distance"
    with st.expander(f"Show the {len(chunks)} retrieved chunks"):
        for i, c in enumerate(chunks, 1):
            st.markdown(f"**Chunk {i}** — page {c['page']}, {score_label} {c['distance']}")
            st.text(c["text"])
            if i < len(chunks):
                st.divider()


def render_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("chunks"))


# ------------------------------------------------------------------- state
st.session_state.setdefault("messages", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("source", indexed_source())

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Document")

    config.DATA_DIR.mkdir(exist_ok=True)
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None:
        chosen = Path(uploaded.name).name
        target = config.DATA_DIR / chosen
        target.write_bytes(uploaded.getbuffer())
    else:
        chosen = None

    use_ocr = st.toggle(
        "Use OCR for scanned pages",
        value=False,
        help=(
            "Off by default for faster indexing. Turn this on to extract text from "
            "scanned or image-only pages; this can take longer."
        ),
    )

    if st.button("Build index", type="primary", disabled=chosen is None, use_container_width=True):
        try:
            with st.status("Indexing…", expanded=True) as status:
                n_pages, n_chunks = run_ingest(config.DATA_DIR / chosen, status, use_ocr)
                status.update(label=f"Indexed {n_pages} pages as {n_chunks} chunks", state="complete")
            st.session_state.source = chosen
            st.session_state.messages = []
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    if st.session_state.source:
        st.success(f"Ready: {st.session_state.source}")
    else:
        st.info("No index yet. Upload a PDF and click Build index.")

    st.header("Retrieval")
    config.USE_RERANKER = st.toggle(
        "Cross-encoder reranker",
        value=config.USE_RERANKER,
        help=f"Fetches {config.FETCH_K} candidates, reranks with {config.RERANKER_MODEL}, "
        f"keeps the best {config.TOP_K}. Slower, but ranks better. "
        "The first use downloads the reranker model.",
    )
    st.caption(
        f"Chunks: {config.CHUNK_SIZE} chars, {config.CHUNK_OVERLAP} overlap · "
        f"Top-K: {config.TOP_K} · LLM: {config.LLM_MODEL}"
    )

    samples = load_sample_questions()
    if samples and st.session_state.source:
        st.header("Sample questions")
        st.caption("From eval/golden_dataset.json (HCLTech report).")
        for i, q in enumerate(samples):
            label = q if len(q) <= 70 else q[:67] + "…"
            if st.button(label, key=f"sample_{i}", use_container_width=True):
                st.session_state.pending = q

    if st.session_state.messages:
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# -------------------------------------------------------------------- main
st.title("Ask your financial report")
st.caption(
    "Answers come only from the indexed PDF and cite page numbers. "
    "Each question is answered on its own; earlier messages are not used as context."
)

for m in st.session_state.messages:
    render_message(m)

has_index = bool(st.session_state.source)
typed = st.chat_input(
    "Ask a question about the report" if has_index else "Build an index in the sidebar first",
    disabled=not has_index,
)
question = st.session_state.pending or typed
st.session_state.pending = None

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the report…"):
                result = answer(question)
            reply = {"role": "assistant", "content": result["answer"], "chunks": result["chunks"]}
        except FileNotFoundError:
            reply = {"role": "assistant", "content": "No index found. Build one from the sidebar.", "chunks": []}
        except Exception as e:
            reply = {
                "role": "assistant",
                "content": f"Something went wrong: `{e}`\n\nCheck that your API key is set in `.env` "
                f"for provider `{config.LLM_PROVIDER}`.",
                "chunks": [],
            }
        st.markdown(reply["content"])
        if reply["content"].strip() == NOT_FOUND and reply["chunks"]:
            st.caption("The retrieved chunks may not contain this. Open them below to check.")
        render_sources(reply["chunks"])
    st.session_state.messages.append(reply)
