"""Cross-encoder reranking: bge-reranker-v2-m3 over (query, chunk) pairs."""

import torch
from sentence_transformers import CrossEncoder

from rag.config import Reranker as RerankerCfg
from rag.types import RetrievedChunk


class Reranker:
    def __init__(self, cfg: RerankerCfg) -> None:
        self.cfg = cfg
        dtype = torch.float16 if cfg.device.startswith("cuda") else torch.float32
        self.model = CrossEncoder(
            cfg.model,
            device=cfg.device,
            max_length=512,
            model_kwargs={"torch_dtype": dtype},
        )

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(
            pairs,
            batch_size=self.cfg.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scored = [
            c.model_copy(update={"score": float(s)}) for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda c: -c.score)
        return scored[: self.cfg.top_k]
