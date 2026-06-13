"""Reference-free retrieval quality proxy — no gold answers needed.

The cross-encoder reranker (bge-reranker-v2-m3) is an independent judge of (query, chunk)
relevance. After hybrid retrieval + fusion, we rerank the candidate pool and read off:

  - rel@1     : sigmoid of the best reranker score  -> "did retrieval surface a relevant
                chunk at all for this query?"
  - rel@k     : mean sigmoid over the top-k kept     -> "how much relevant context?"
  - answerable: fraction of queries with rel@1 >= threshold (retrieval coverage)

Because the reranker is a different model from the embedder, using it to judge what the
embedder surfaced is not circular. Use this to A/B the embedder, top_k, fusion and
chunking WITHOUT any reference answers. Trust deltas between configs.

Usage:
    uv run python scripts/eval_retrieval.py --n 300
    uv run python scripts/eval_retrieval.py --n 300 --threshold 0.5 --out artifacts/retr.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Must precede torch/bm25s import on macOS (see scripts/answer.py).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from rag.chunking import load_chunks  # noqa: E402
from rag.config import load_config, seed_everything  # noqa: E402
from rag.eval import is_no_data_answer  # noqa: E402
from rag.rerank import Reranker  # noqa: E402
from rag.retrieval.bm25 import BM25Retriever  # noqa: E402
from rag.retrieval.dense import DenseRetriever  # noqa: E402
from rag.retrieval.hybrid import HybridRetriever  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300, help="evaluate the first N questions")
    p.add_argument("--threshold", type=float, default=0.5, help="rel@1 cutoff for 'answerable'")
    p.add_argument("--out", default=None, help="optional per-question breakdown CSV")
    p.add_argument(
        "--device", default=None, help="override embedder/reranker device (e.g. cpu, cuda)"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    seed_everything(cfg.seed)
    if args.device:
        cfg.embedder.device = args.device
        cfg.reranker.device = args.device

    print("[retr] loading chunks + indexes…")
    chunks = load_chunks(cfg.resolve(cfg.paths.chunks_parquet))
    dense = DenseRetriever(cfg.embedder)
    dense.load(cfg.resolve(cfg.paths.faiss_index))
    bm25 = BM25Retriever(cfg.bm25)
    bm25.load(cfg.resolve(cfg.paths.bm25_pickle))
    hybrid = HybridRetriever(dense, bm25, cfg.retrieval, chunks)
    print("[retr] loading reranker…")
    reranker = Reranker(cfg.reranker)

    questions = pl.read_csv(cfg.resolve(cfg.paths.questions_csv), infer_schema_length=0)
    if args.n is not None:
        questions = questions.head(args.n)
    rows = list(questions.iter_rows(named=True))
    queries = [r["query"] for r in rows]

    print(f"[retr] retrieving {len(queries)} queries…")
    candidate_lists = hybrid.search(queries)

    rel1, relk, empties = [], [], 0
    per_q = []
    for row, cands in tqdm(zip(rows, candidate_lists), total=len(rows), desc="rerank"):
        q_id = str(row["q_id"])
        if not cands:
            empties += 1
            rel1.append(0.0)
            relk.append(0.0)
            per_q.append({"q_id": q_id, "rel@1": 0.0, "rel@k": 0.0, "n_cand": 0})
            continue
        reranked = reranker.rerank(row["query"], cands)
        scores = _sigmoid(np.array([c.score for c in reranked], dtype=np.float64))
        r1, rk = float(scores[0]), float(scores.mean())
        rel1.append(r1)
        relk.append(rk)
        per_q.append({"q_id": q_id, "rel@1": r1, "rel@k": rk, "n_cand": len(cands)})

    rel1 = np.array(rel1)
    relk = np.array(relk)
    answerable = float((rel1 >= args.threshold).mean())
    print(
        f"\n[retr] n={len(rows)}  (reranker={cfg.reranker.model}, top_k={cfg.reranker.top_k}, "
        f"embedder={cfg.embedder.model})\n"
        f"  rel@1   mean={rel1.mean():.4f}  median={np.median(rel1):.4f}\n"
        f"  rel@k   mean={relk.mean():.4f}\n"
        f"  answerable (rel@1>={args.threshold}): {answerable:.1%}\n"
        f"  empty retrieval: {empties}/{len(rows)}"
    )

    # Gate viability: do "Нет ответа" reference questions have LOWER rel@1 than substantive
    # ones? If so, a reranker-score gate cleanly abstains on them. Label from sample_submission.
    ref_path = cfg.resolve(cfg.paths.sample_submission_csv)
    refs = pl.read_csv(ref_path, infer_schema_length=0)
    acol = "answer_new" if "answer_new" in refs.columns else "answer"
    nodata = {
        str(r["q_id"])
        for r in refs.select(pl.col("q_id").cast(pl.Utf8), acol).iter_rows(named=True)
        if is_no_data_answer(r[acol] or "")
    }
    qids = [str(r["q_id"]) for r in rows]
    is_nd = np.array([q in nodata for q in qids])
    if is_nd.any() and (~is_nd).any():
        nd_r1, sub_r1 = rel1[is_nd], rel1[~is_nd]
        print(
            f"\n[retr] gate viability (ref = 'Нет ответа' vs substantive):\n"
            f"  no-data refs    n={is_nd.sum():3d}  rel@1 mean={nd_r1.mean():.4f}  median={np.median(nd_r1):.4f}\n"
            f"  substantive refs n={(~is_nd).sum():3d}  rel@1 mean={sub_r1.mean():.4f}  median={np.median(sub_r1):.4f}\n"
            f"  (big gap → a rel@1 threshold can separate them → reranker gate works)\n"
            f"  gate sweep: abstain when rel@1 < T\n"
            f"  {'T':>6} {'no-data caught':>16} {'substantive lost':>18}"
        )
        for t in (0.55, 0.60, 0.65, 0.70, 0.75):
            caught = float((nd_r1 < t).mean())   # good: we abstain on no-data
            lost = float((sub_r1 < t).mean())     # bad: we abstain on answerable
            print(f"  {t:>6.2f} {caught:>15.0%} {lost:>17.0%}")

    if args.out:
        out = cfg.resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(per_q).with_columns(
            pl.col("q_id").is_in(list(nodata)).alias("ref_is_no_data")
        ).write_csv(out)
        print(f"[retr] per-question breakdown → {out}")


if __name__ == "__main__":
    main()
