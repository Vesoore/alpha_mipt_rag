"""Dense retrieval: multilingual-e5 + FAISS IndexFlatIP on L2-normed vectors."""

from pathlib import Path

import faiss
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

from rag.config import Embedder


class DenseRetriever:
    def __init__(self, cfg: Embedder, model: SentenceTransformer | None = None) -> None:
        self.cfg = cfg
        self.model = (
            model
            if model is not None
            else SentenceTransformer(cfg.model, device=cfg.device)
        )
        self.index: faiss.Index | None = None
        self.dim: int | None = None

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.cfg.passage_prefix + t for t in texts]
        emb = self.model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        return emb.astype(np.float32, copy=False)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        prefixed = [self.cfg.query_prefix + q for q in queries]
        emb = self.model.encode(
            prefixed,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32, copy=False)

    def build_index(self, chunks: pl.DataFrame) -> None:
        emb = self.embed_passages(chunks["text"].to_list())
        self.dim = emb.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(emb)

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise RuntimeError("call build_index() first")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path) -> None:
        self.index = faiss.read_index(str(path))
        self.dim = self.index.d

    def search(self, queries: list[str], top_n: int) -> list[list[tuple[int, float]]]:
        if self.index is None:
            raise RuntimeError("index not loaded")
        emb = self.embed_queries(queries)
        scores, idxs = self.index.search(emb, top_n)
        out = []
        for s_row, i_row in zip(scores, idxs):
            out.append([(int(i), float(s)) for i, s in zip(i_row, s_row) if i >= 0])
        return out
