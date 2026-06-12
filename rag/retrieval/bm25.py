"""BM25 retrieval over chunks via the bm25s library.

Tokenization (lowercase + stopword removal + stemming) must be identical at index
time and query time — the stored vocabulary is stemmed, so a query stemmed differently
would miss it. Both paths go through `BM25Retriever._tokenize`, and the stemmer/stopwords
are rebuilt from config in `__init__` (they are not pickled with the index), so a loaded
retriever must be constructed with the same `cfg.bm25` that built the index.
"""

import pickle
from collections.abc import Callable
from pathlib import Path

import bm25s
import polars as pl

from rag.config import BM25 as BM25Cfg


def _make_stemmer(language: str | None) -> Callable | None:
    if not language:
        return None
    import Stemmer  # PyStemmer

    return Stemmer.Stemmer(language).stemWords


class BM25Retriever:
    def __init__(self, cfg: BM25Cfg | None = None) -> None:
        self.bm25: bm25s.BM25 | None = None
        cfg = cfg or BM25Cfg()
        self.stopwords: str | None = cfg.stopwords or None
        self.stemmer = _make_stemmer(cfg.stemmer_language)

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        return bm25s.tokenize(
            texts,
            stopwords=self.stopwords,
            stemmer=self.stemmer,
            return_ids=False,
            show_progress=False,
        )

    def build_index(self, chunks: pl.DataFrame) -> None:
        tokens = self._tokenize(chunks["text"].to_list())
        self.bm25 = bm25s.BM25()
        self.bm25.index(tokens)

    def save(self, path: str | Path) -> None:
        if self.bm25 is None:
            raise RuntimeError("call build_index() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, queries: list[str], top_n: int) -> list[list[tuple[int, float]]]:
        if self.bm25 is None:
            raise RuntimeError("index not loaded")
        q_tokens = self._tokenize(queries)
        results, scores = self.bm25.retrieve(q_tokens, k=top_n)
        out = []
        for r_row, s_row in zip(results, scores):
            out.append([(int(i), float(s)) for i, s in zip(r_row, s_row)])
        return out
