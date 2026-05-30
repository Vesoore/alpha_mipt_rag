"""Load websites.csv and clean it for retrieval.

Decisions driven by the data-corpus-shape memory (profiled 2026-05-30):
- 17 duplicate URLs and 16 exact-duplicate texts: drop by text equality, keep
  the shortest URL per cluster.
- City-template near-duplicates (same body, different city in title): collapse
  by hash(text[:500]), keep one representative per cluster.
- A handful of corrupt PDFs / 10-char stubs: drop docs with <50 chars after
  control-char strip.
- alfabank.by and private.auth.alfabank.ru are out of scope: filter by host.
"""

import hashlib
import re
from pathlib import Path

import polars as pl

from rag.config import Cleaning

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RUN_RE = re.compile(r"[ \t]{2,}")
NEWLINE_RUN_RE = re.compile(r"\n{3,}")


def _clean_text(s: str | None) -> str:
    if not s:
        return ""
    s = CONTROL_CHARS_RE.sub("", s)
    s = WHITESPACE_RUN_RE.sub(" ", s)
    s = NEWLINE_RUN_RE.sub("\n\n", s)
    return s.strip()


def _host(url: str | None) -> str:
    if not url:
        return ""
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1) if m else ""


def _prefix_hash(text: str, n_chars: int) -> str:
    return hashlib.blake2s(text[:n_chars].encode("utf-8"), digest_size=8).hexdigest()


def load_and_clean(websites_csv: str | Path, cfg: Cleaning) -> pl.DataFrame:
    """Return a cleaned, deduped DataFrame with columns: web_id, url, kind, title, text."""
    df = pl.read_csv(websites_csv, infer_schema_length=0)
    n0 = df.height

    df = df.with_columns(
        pl.col("text").map_elements(_clean_text, return_dtype=pl.String).alias("text"),
        pl.col("title").fill_null("").alias("title"),
        pl.col("url").fill_null("").alias("url"),
    )

    df = df.filter(pl.col("text").str.len_chars() >= cfg.min_chars)
    n1 = df.height

    df = df.with_columns(
        pl.col("url").map_elements(_host, return_dtype=pl.String).alias("host"),
    )
    df = df.filter(~pl.col("host").is_in(cfg.drop_hosts))
    n2 = df.height

    df = df.sort(pl.col("url").str.len_chars()).unique(
        subset=["text"], keep="first", maintain_order=True
    )
    n3 = df.height

    df = df.with_columns(
        pl.col("text")
        .map_elements(
            lambda t: _prefix_hash(t, cfg.dedup_prefix_chars), return_dtype=pl.String
        )
        .alias("_prefix_hash"),
    )
    df = df.unique(subset=["_prefix_hash"], keep="first", maintain_order=True).drop(
        "_prefix_hash"
    )
    n4 = df.height

    print(f"[data] rows: {n0} → filter<{cfg.min_chars}ch → {n1} → host-filter → {n2}")
    print(
        f"[data]       → exact-text dedup → {n3} → near-dup({cfg.dedup_prefix_chars}ch) → {n4}"
    )
    return df.select(["web_id", "url", "kind", "title", "text"])
