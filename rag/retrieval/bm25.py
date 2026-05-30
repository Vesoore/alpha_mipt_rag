"""BM25 retrieval over chunks via the bm25s library."""

import pickle
from pathlib import Path

import bm25s
import polars as pl


def _tokenize(texts: list[str]) -> list[list[str]]:
    # bm25s.tokenize returns Tokenized; passing return_ids=False gives plain token lists.
    return bm25s.tokenize(texts, stopwords=None, stemmer=None, return_ids=False)


class BM25Retriever:
    def __init__(self) -> None:
        self.bm25: bm25s.BM25 | None = None

    def build_index(self, chunks: pl.DataFrame) -> None:
        tokens = _tokenize(chunks["text"].to_list())
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
        q_tokens = _tokenize(queries)
        results, scores = self.bm25.retrieve(q_tokens, k=top_n)
        out = []
        for r_row, s_row in zip(results, scores):
            out.append([(int(i), float(s)) for i, s in zip(r_row, s_row)])
        return out
