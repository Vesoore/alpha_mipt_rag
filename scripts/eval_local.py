"""Score a submission CSV locally with Recall-L, against reference answers.

The references are the `answer_new` column of `data/sample_submission.csv` (a usable
dev set). Use this BEFORE spending one of the 3 daily uploads, and to tune the length
budget / top_k / chunking by the offline delta.

Usage:
    uv run python scripts/eval_local.py --pred submission.csv
    uv run python scripts/eval_local.py --pred submissions/run_n20.csv --n 200
    uv run python scripts/eval_local.py --pred submission.csv \
        --ref data/sample_submission.csv --out artifacts/eval_per_q.csv --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl  # noqa: E402

from rag.config import load_config  # noqa: E402
from rag.eval import recall_l, strip_rag_artifacts  # noqa: E402


def _answer_col(df: pl.DataFrame) -> str:
    for c in ("answer", "answer_new"):
        if c in df.columns:
            return c
    raise ValueError(f"no answer column in {df.columns} (expected 'answer' or 'answer_new')")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="submission CSV to score")
    p.add_argument("--ref", default=None, help="reference CSV (default: cfg sample_submission)")
    p.add_argument("--n", type=int, default=None, help="score only the first N joined questions")
    p.add_argument("--out", default=None, help="optional path to dump the per-question breakdown")
    p.add_argument("--device", default=None, help="torch device for BERTScore (cuda/cpu)")
    p.add_argument(
        "--raw-refs",
        action="store_true",
        help="score against raw references (skip stripping RAG-artifact boilerplate like "
        "'Согласно Фрагменту N'); default is to clean them",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    ec = cfg.eval

    ref_path = args.ref or cfg.resolve(cfg.paths.sample_submission_csv)
    pred = pl.read_csv(args.pred, infer_schema_length=0)
    ref = pl.read_csv(ref_path, infer_schema_length=0)

    pred = pred.select(pl.col("q_id").cast(pl.Utf8), pl.col(_answer_col(pred)).alias("pred"))
    ref = ref.select(pl.col("q_id").cast(pl.Utf8), pl.col(_answer_col(ref)).alias("ref"))

    joined = pred.join(ref, on="q_id", how="inner")
    if joined.height == 0:
        raise SystemExit("no overlapping q_id between prediction and reference")
    if joined.height < pred.height:
        print(f"[eval] warning: {pred.height - joined.height} predicted q_id(s) had no reference")
    if args.n is not None:
        joined = joined.head(args.n)

    predictions = joined["pred"].fill_null("").to_list()
    references = joined["ref"].fill_null("").to_list()
    if not args.raw_refs:
        # The references are a baseline RAG's output and carry citation boilerplate that
        # distorts recall/length; clean both sides so the comparison is apples-to-apples.
        n_before = sum(strip_rag_artifacts(r) != r for r in references)
        references = [strip_rag_artifacts(r) for r in references]
        predictions = [strip_rag_artifacts(p) for p in predictions]
        print(f"[eval] cleaned RAG artifacts from {n_before} references (use --raw-refs to skip)")

    print(f"[eval] scoring {joined.height} questions (BERTScore on {args.device or 'auto'})…")
    result = recall_l(
        q_ids=joined["q_id"].to_list(),
        predictions=predictions,
        references=references,
        lang=ec.bertscore_lang,
        model_type=ec.bertscore_model,
        num_layers=ec.bertscore_num_layers,
        length_tokenizer=ec.length_tokenizer,
        batch_size=ec.batch_size,
        device=args.device,
        idf=ec.idf,
    )

    print(result.summary())
    worst = result.per_question.sort("recall_l").head(5)
    print("\n[eval] 5 worst questions (q_id, recall_l, l_a/l_r):")
    for row in worst.iter_rows(named=True):
        print(f"  {row['q_id']}: {row['recall_l']:.3f}  ({row['l_a']}/{row['l_r']} tok)")

    if args.out:
        out = cfg.resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.per_question.write_csv(out)
        print(f"\n[eval] per-question breakdown → {out}")


if __name__ == "__main__":
    main()
