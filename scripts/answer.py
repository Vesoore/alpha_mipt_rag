"""Load index + models, answer every question, write submission CSV.

Usage:
    uv run python scripts/answer.py                # full run on all questions
    uv run python scripts/answer.py --n 20         # smoke test on first 20
    uv run python scripts/answer.py --n 20 --out submissions/smoke.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# See scripts/build_index.py for why these must be set before any imports.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from rag.chunking import load_chunks  # noqa: E402
from rag.config import load_config, seed_everything  # noqa: E402
from rag.generation import Generator  # noqa: E402
from rag.pipeline import Pipeline  # noqa: E402
from rag.rerank import Reranker  # noqa: E402
from rag.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag.retrieval.dense import DenseRetriever  # noqa: E402
from rag.length import trim_answer  # noqa: E402
from rag.submission import write_submission  # noqa: E402
from rag.types import Answer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=None, help="answer only the first N questions (smoke)")
    p.add_argument("--out", type=str, default=None, help="override submission output path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    seed_everything(cfg.seed)

    print("[answer] loading chunks + indexes…")
    chunks = load_chunks(cfg.resolve(cfg.paths.chunks_parquet))

    dense = DenseRetriever(cfg.embedder)
    dense.load(cfg.resolve(cfg.paths.faiss_index))

    bm25 = BM25Retriever()
    bm25.load(cfg.resolve(cfg.paths.bm25_pickle))

    print("[answer] loading reranker…")
    reranker = Reranker(cfg.reranker)

    print(f"[answer] loading generator ({cfg.generator.backend})…")
    generator = Generator(cfg.generator, seed=cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.chunking.tokenizer_model)

    pipeline = Pipeline(cfg, chunks, dense, bm25, reranker, generator, tokenizer)

    questions = pl.read_csv(cfg.resolve(cfg.paths.questions_csv), infer_schema_length=0)
    if args.n is not None:
        questions = questions.head(args.n)

    print(f"[answer] running retrieval on {questions.height} questions…")
    t0 = time.time()
    rows = list(questions.iter_rows(named=True))
    no_data = cfg.length.no_data_phrase

    # Phase 1: retrieval + rerank + grounding (sequential, per-question)
    contexts = []
    no_data_ids: list[str] = []
    for row in tqdm(rows, desc="retrieve"):
        ctx = pipeline.retrieve_and_ground(str(row["q_id"]), row["query"])
        if ctx is None:
            no_data_ids.append(str(row["q_id"]))
        else:
            contexts.append(ctx)

    # Phase 2: batch generation (all prompts in one vllm call)
    print(f"[answer] generating {len(contexts)} answers (batch)…")
    raw_answers = pipeline.generator.generate_batch(contexts)

    answers = []
    for ctx, raw in zip(contexts, raw_answers):
        text = raw.strip() or no_data
        answers.append(Answer(q_id=ctx.q_id, answer=trim_answer(text, cfg.length.answer_max_chars)))
    for q_id in no_data_ids:
        answers.append(Answer(q_id=q_id, answer=no_data))

    dt = time.time() - t0
    print(f"[answer] done in {dt:.1f}s ({dt / max(len(answers), 1):.2f}s/q)")

    out_path = args.out or cfg.paths.submission_csv
    write_submission(
        answers,
        sample_path=cfg.resolve(cfg.paths.sample_submission_csv),
        out_path=cfg.resolve(out_path),
    )


if __name__ == "__main__":
    main()
