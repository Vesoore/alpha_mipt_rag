"""Build retrieval artifacts: clean → chunk → embed → BM25 → save.

Usage:
    uv run python scripts/build_index.py
"""

import os
import sys
from pathlib import Path

# Must be set BEFORE torch/numpy/bm25s import: otherwise torch's libomp and
# bm25s' loky pool clash on macOS and the process segfaults at BM25 indexing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.chunking import chunk_dataframe, save_chunks  # noqa: E402
from rag.config import load_config, seed_everything  # noqa: E402
from rag.data import load_and_clean  # noqa: E402
from rag.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag.retrieval.dense import DenseRetriever  # noqa: E402


def main() -> None:
    cfg = load_config()
    seed_everything(cfg.seed)

    docs = load_and_clean(cfg.resolve(cfg.paths.websites_csv), cfg.cleaning)
    chunks = chunk_dataframe(docs, cfg.chunking)
    save_chunks(chunks, cfg.resolve(cfg.paths.chunks_parquet))

    print("[index] building dense FAISS index…")
    dense = DenseRetriever(cfg.embedder)
    dense.build_index(chunks)
    dense.save(cfg.resolve(cfg.paths.faiss_index))
    print(f"[index] dense: {chunks.height} vectors, dim={dense.dim}")

    print("[index] building BM25 index…")
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    bm25.save(cfg.resolve(cfg.paths.bm25_pickle))
    print(f"[index] bm25: {chunks.height} docs")

    print("[index] done.")


if __name__ == "__main__":
    main()
