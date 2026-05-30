"""Hybrid retrieval: dense ⊕ BM25 fused with Reciprocal Rank Fusion."""

import polars as pl

from rag.config import Retrieval
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.dense import DenseRetriever
from rag.types import RetrievedChunk


def _rrf(
    ranked_lists: list[list[tuple[int, float]]],
    k: int,
    top_n: int,
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])[:top_n]


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        cfg: Retrieval,
        chunks: pl.DataFrame,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.cfg = cfg
        self.chunks = chunks

    def search(self, queries: list[str]) -> list[list[RetrievedChunk]]:
        d_results = self.dense.search(queries, self.cfg.dense_top_n)
        b_results = self.bm25.search(queries, self.cfg.bm25_top_n)
        out: list[list[RetrievedChunk]] = []
        for d, b in zip(d_results, b_results):
            fused = _rrf([d, b], k=self.cfg.rrf_k, top_n=self.cfg.fused_top_n)
            chunks = [self._row_to_chunk(idx, score) for idx, score in fused]
            out.append(chunks)
        return out

    def _row_to_chunk(self, idx: int, score: float) -> RetrievedChunk:
        row = self.chunks.row(idx, named=True)
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            web_id=row["web_id"],
            url=row["url"],
            title=row.get("title", ""),
            text=row["text"],
            score=score,
        )
