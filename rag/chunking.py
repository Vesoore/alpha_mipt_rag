"""Paragraph-aware chunker using the e5 tokenizer for length accounting.

Strategy:
1. Split on blank lines into paragraphs.
2. If a paragraph fits in `target_tokens`, treat it as a unit.
3. If not, sub-split into sentences. If a sentence still doesn't fit,
   hard-cut on tokens (rare — only happens in malformed PDFs).
4. Greedy-pack units into chunks ≤ ~`target_tokens`, carrying the last
   `overlap_tokens` from each emitted chunk forward for continuity.

Mega-PDFs (web_id 1704/1705, ~1.47M chars each) are handled by streaming
paragraphs through a generator — the full doc is never tokenized at once.
"""

import re
from collections.abc import Iterator
from pathlib import Path

import polars as pl
from tqdm import tqdm
from transformers import AutoTokenizer

from rag.config import Chunking

PARAGRAPH_RE = re.compile(r"\n\s*\n")
SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")


def _split_paragraphs(text: str) -> Iterator[str]:
    for p in PARAGRAPH_RE.split(text):
        p = p.strip()
        if p:
            yield p


def _split_sentences(paragraph: str) -> Iterator[str]:
    for s in SENTENCE_RE.split(paragraph):
        s = s.strip()
        if s:
            yield s


def _units(text: str, target: int, tokenizer) -> Iterator[list[int]]:
    """Yield token-id lists, each ≤ target tokens."""
    for para in _split_paragraphs(text):
        ids = tokenizer.encode(para, add_special_tokens=False)
        if len(ids) <= target:
            yield ids
            continue
        for sent in _split_sentences(para):
            sids = tokenizer.encode(sent, add_special_tokens=False)
            if len(sids) <= target:
                yield sids
                continue
            for i in range(0, len(sids), target):
                yield sids[i : i + target]


def chunk_text(
    text: str,
    target: int,
    overlap: int,
    tokenizer,
) -> Iterator[str]:
    buffer: list[int] = []
    for unit in _units(text, target, tokenizer):
        if buffer and len(buffer) + len(unit) > target:
            yield tokenizer.decode(buffer, skip_special_tokens=True)
            buffer = buffer[-overlap:] if overlap > 0 else []
        buffer.extend(unit)
    if buffer:
        yield tokenizer.decode(buffer, skip_special_tokens=True)


def chunk_dataframe(
    df: pl.DataFrame,
    cfg: Chunking,
) -> pl.DataFrame:
    """Chunk every doc in `df`. Returns rows: chunk_id, web_id, url, title, text, n_tokens."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model)

    rows: list[dict] = []
    for row in tqdm(df.iter_rows(named=True), total=df.height, desc="chunking"):
        web_id = str(row["web_id"])
        url = row["url"] or ""
        title = row["title"] or ""
        text = row["text"] or ""
        for idx, piece in enumerate(
            chunk_text(text, cfg.target_tokens, cfg.overlap_tokens, tokenizer)
        ):
            n_tokens = len(tokenizer.encode(piece, add_special_tokens=False))
            rows.append(
                {
                    "chunk_id": f"{web_id}::{idx:04d}",
                    "web_id": web_id,
                    "url": url,
                    "title": title,
                    "text": piece,
                    "n_tokens": n_tokens,
                }
            )
    chunks = pl.DataFrame(rows)
    print(
        f"[chunk] {df.height} docs → {chunks.height} chunks "
        f"(mean tokens={chunks['n_tokens'].mean():.0f}, "
        f"p95={chunks['n_tokens'].quantile(0.95):.0f})"
    )
    return chunks


def save_chunks(chunks: pl.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    chunks.write_parquet(path)


def load_chunks(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)
