"""End-to-end pipeline: q_id, query → grounded Answer."""

import polars as pl

from rag.config import Config
from rag.generation import Generator
from rag.grounding import assemble_context
from rag.length import trim_answer
from rag.rerank import Reranker
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.types import Answer, GroundingContext


class Pipeline:
    def __init__(
        self,
        cfg: Config,
        chunks: pl.DataFrame,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        reranker: Reranker,
        generator: Generator,
        tokenizer,
    ) -> None:
        self.cfg = cfg
        self.hybrid = HybridRetriever(dense, bm25, cfg.retrieval, chunks)
        self.reranker = reranker
        self.generator = generator
        self.tokenizer = tokenizer

    def retrieve_and_ground(self, q_id: str, query: str) -> GroundingContext | None:
        """Retrieval + rerank + context assembly; returns None when no candidates found."""
        candidates = self.hybrid.search([query])[0]
        if not candidates:
            return None
        reranked = self.reranker.rerank(query, candidates)
        ctx = assemble_context(
            q_id=q_id,
            query=query,
            chunks=reranked,
            cfg=self.cfg.grounding,
            tokenizer=self.tokenizer,
        )
        return ctx if ctx.chunks else None

    def answer(self, q_id: str, query: str) -> Answer:
        ctx = self.retrieve_and_ground(q_id, query)
        if ctx is None:
            return Answer(q_id=q_id, answer=self.cfg.length.no_data_phrase)
        raw = self.generator.generate(ctx)
        if not raw.strip():
            raw = self.cfg.length.no_data_phrase
        trimmed = trim_answer(raw, self.cfg.length.answer_max_chars)
        return Answer(q_id=q_id, answer=trimmed)
