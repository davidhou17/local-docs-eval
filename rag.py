"""Zero-dependency local RAG: chunk docs, embed via Ollama, retrieve by cosine similarity.

Stdlib only -- uses urllib for Ollama /api/embed, json for index persistence, math for cosine sim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "mxbai-embed-large"
CHUNK_TARGET_CHARS = 500
CHUNK_OVERLAP_CHARS = 100
TOP_K = 5


# ---------------------------------------------------------------------------
# Text loading & chunking
# ---------------------------------------------------------------------------

def _strip_frontmatter(text: str) -> tuple[str, str]:
    """Strip YAML frontmatter (---...---) and return (title, body)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return ("", text)
    fm = m.group(1)
    title = ""
    for line in fm.splitlines():
        if line.strip().startswith("title:"):
            title = line.split(":", 1)[1].strip().strip('"').strip("'")
            break
    return (title, text[m.end():])


def load_docs(docs_dir: str) -> list[dict[str, str]]:
    """Read all .mdx files under docs_dir. Returns list of {path, title, text}."""
    docs_path = Path(docs_dir)
    results = []
    for mdx in sorted(docs_path.rglob("*.mdx")):
        raw = mdx.read_text(encoding="utf-8", errors="replace")
        title, body = _strip_frontmatter(raw)
        rel = str(mdx.relative_to(docs_path))
        results.append({"path": rel, "title": title, "text": body.strip()})
    return results


def chunk_docs(docs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Split docs into overlapping chunks. Each chunk has {text, source, title}."""
    chunks: list[dict[str, str]] = []
    for doc in docs:
        doc_chunks = _chunk_text(doc["text"])
        for chunk in doc_chunks:
            chunks.append({
                "text": chunk,
                "source": doc["path"],
                "title": doc["title"],
            })
    return chunks


def _hard_split(text: str, limit: int) -> list[str]:
    """Split text that exceeds *limit* chars on sentence/line boundaries, falling back to a hard cut."""
    pieces: list[str] = []
    while len(text) > limit:
        cut = text.rfind(". ", 0, limit)
        if cut == -1:
            cut = text.rfind("\n", 0, limit)
        if cut == -1 or cut < limit // 4:
            cut = limit
        pieces.append(text[: cut + 1].strip())
        text = text[cut + 1 :].strip()
    if text.strip():
        pieces.append(text.strip())
    return pieces


def _chunk_text(text: str) -> list[str]:
    """Split text into chunks of ~CHUNK_TARGET_CHARS with overlap, splitting on section/paragraph boundaries."""
    sections = re.split(r"\n(?=##\s)", text)

    paragraphs: list[str] = []
    for section in sections:
        parts = section.split("\n\n")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) > CHUNK_TARGET_CHARS:
                paragraphs.extend(_hard_split(p, CHUNK_TARGET_CHARS))
            else:
                paragraphs.append(p)

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > CHUNK_TARGET_CHARS:
            chunks.append(current.strip())
            overlap_start = max(0, len(current) - CHUNK_OVERLAP_CHARS)
            current = current[overlap_start:].strip() + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks if chunks else [text[:CHUNK_TARGET_CHARS]] if text.strip() else []


# ---------------------------------------------------------------------------
# Ollama embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str], model: str = DEFAULT_EMBED_MODEL, batch_size: int = 32) -> list[list[float]]:
    """Embed a list of texts via Ollama /api/embed. Returns list of embedding vectors."""
    base_url = (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL).rstrip("/")

    cleaned = [t if t.strip() else " " for t in texts]
    all_embeddings: list[list[float]] = []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i : i + batch_size]
        body: dict[str, Any] = {
            "model": model,
            "input": batch,
        }
        req = urllib.request.Request(
            f"{base_url}/api/embed",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"Ollama embed HTTP {e.code} for batch {i // batch_size + 1} "
                f"({len(batch)} texts, model '{model}'). "
                f"Response: {detail or e.reason}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama embed failed (is Ollama running at {base_url}?). "
                f"Try: ollama pull {model}\nError: {e}"
            ) from e

        embeddings = data.get("embeddings")
        if not embeddings:
            raise RuntimeError(f"Ollama /api/embed returned no embeddings: {data}")
        all_embeddings.extend(embeddings)

    return all_embeddings


# ---------------------------------------------------------------------------
# Vector index: build, save, load
# ---------------------------------------------------------------------------

def _docs_fingerprint(docs_dir: str) -> str:
    """Fast fingerprint of the docs directory: hash of sorted (path, mtime, size) tuples."""
    entries = []
    for mdx in sorted(Path(docs_dir).rglob("*.mdx")):
        stat = mdx.stat()
        entries.append(f"{mdx}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def _index_path(docs_dir: str, embed_model: str) -> Path:
    """Where to cache the index JSON."""
    safe_model = embed_model.replace("/", "_").replace(":", "_")
    return Path(docs_dir) / f".rag_index_{safe_model}.json"


def build_index(
    docs_dir: str,
    embed_model: str = DEFAULT_EMBED_MODEL,
    force: bool = False,
) -> list[dict]:
    """Build (or load cached) RAG index. Returns list of {text, source, title, embedding}."""
    idx_file = _index_path(docs_dir, embed_model)
    fingerprint = _docs_fingerprint(docs_dir)

    if not force and idx_file.exists():
        try:
            cached = json.loads(idx_file.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("embed_model") == embed_model:
                print(f"  Using cached index ({len(cached['chunks'])} chunks) from {idx_file.name}")
                return cached["chunks"]
        except (json.JSONDecodeError, KeyError):
            pass

    print(f"  Building RAG index from {docs_dir} ...")
    docs = load_docs(docs_dir)
    print(f"  Loaded {len(docs)} documents")

    chunks = chunk_docs(docs)
    print(f"  Created {len(chunks)} chunks, embedding with {embed_model} ...")

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, model=embed_model)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    idx_file.write_text(
        json.dumps({
            "fingerprint": fingerprint,
            "embed_model": embed_model,
            "chunks": chunks,
        }),
        encoding="utf-8",
    )
    print(f"  Index cached to {idx_file.name}")
    return chunks


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(
    query: str,
    index: list[dict],
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = TOP_K,
) -> list[dict]:
    """Embed query, return top_k most similar chunks from the index."""
    query_emb = embed_texts([query], model=embed_model)[0]
    scored = []
    for chunk in index:
        sim = _cosine_similarity(query_emb, chunk["embedding"])
        scored.append((sim, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]
