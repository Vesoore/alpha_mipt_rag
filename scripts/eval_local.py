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
from rag.eval import is_no_data_answer, recall_l, strip_rag_artifacts  # noqa: E402


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
        "--worst", type=int, default=8, help="show this many worst questions with query/answer/ref text"
    )
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

    # Split by reference type. Comparing a substantive answer to a "Нет ответа" reference
    # measures no-data detection, not answer quality — so report the two separately, else
    # the baseline's abstentions silently drag the headline number down.
    pq = result.per_question
    nodata_ids = {
        qid for qid, r in zip(joined["q_id"].to_list(), references) if is_no_data_answer(r)
    }
    answered_set = {qid for qid, p in zip(joined["q_id"].to_list(), predictions) if not is_no_data_answer(p)}
    sub = pq.filter(~pl.col("q_id").is_in(list(nodata_ids)))          # substantive references
    nod = pq.filter(pl.col("q_id").is_in(list(nodata_ids)))           # "Нет ответа" references
    print(
        f"\n[eval] reference buckets:\n"
        f"  substantive refs: {sub.height}/{pq.height}  →  Recall-L = {sub['recall_l'].mean():.4f} "
        f"(R_BERT={sub['bert_recall'].mean():.4f})   ← your real answer quality"
    )
    if nod.height:
        # On no-data refs the metric rewards abstaining. How often did we wrongly answer?
        over_answered = sum(1 for qid in nodata_ids if qid in answered_set)
        print(
            f"  no-data refs:     {nod.height}/{pq.height}  →  Recall-L = {nod['recall_l'].mean():.4f}; "
            f"you ANSWERED on {over_answered}/{nod.height} of them "
            f"({over_answered / nod.height:.0%}) instead of abstaining"
        )
        print(
            "  NOTE: high over-answer here is only a problem if those answers are NOT grounded "
            "in retrieval — check rel@1 (eval_retrieval) before forcing abstention."
        )

    # Attach query + scored pred/ref text so the worst cases are actually diagnosable:
    # is a zeroed question a no-data case (ref is a short "Нет ответа") or just too long?
    questions = pl.read_csv(cfg.resolve(cfg.paths.questions_csv), infer_schema_length=0).select(
        pl.col("q_id").cast(pl.Utf8), pl.col("query")
    )
    scored = pl.DataFrame({"q_id": joined["q_id"], "pred": predictions, "ref": references})
    diag = result.per_question.join(scored, on="q_id", how="left").join(
        questions, on="q_id", how="left"
    )

    def _trunc(s: str, n: int = 140) -> str:
        s = " ".join((s or "").split())
        return s if len(s) <= n else s[:n] + "…"

    worst = diag.sort("recall_l").head(args.worst)
    print(f"\n[eval] {args.worst} worst questions (recall_l | l_a/l_r tok):")
    for row in worst.iter_rows(named=True):
        print(
            f"\n  q{row['q_id']}  recall_l={row['recall_l']:.3f}  "
            f"R={row['bert_recall']:.2f}  L={row['length_penalty']:.2f}  "
            f"({row['l_a']}/{row['l_r']} tok)"
        )
        print(f"    Q: {_trunc(row['query'])}")
        print(f"    A: {_trunc(row['pred'])}")
        print(f"    R: {_trunc(row['ref'])}")

    if args.out:
        out = cfg.resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        diag.write_csv(out)
        print(f"\n[eval] per-question breakdown (with query/pred/ref) → {out}")


if __name__ == "__main__":
    main()
