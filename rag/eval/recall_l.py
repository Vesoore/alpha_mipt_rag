"""Local Recall-L scorer — the offline proxy for the leaderboard metric.

`Recall_L(q) = R_BERT(q) · L(q)`, averaged over questions, where `R_BERT` is the
recall component of multilingual BERTScore and `L(q)` is the piecewise length
penalty from CLAUDE.md ("Metric").

This lets us tune length budget / top_k / chunking offline against the reference
answers in `data/sample_submission.csv` instead of burning daily submissions.

Caveat: the platform's exact BERTScore model and the tokenizer it counts length
with are not published. We default to `bert-base-multilingual-cased` (what
bert-score uses for `lang="ru"`) for both, and keep both configurable so the
local number tracks the leaderboard as closely as we can verify. Treat absolute
values as a proxy; trust the *delta* between configs.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


def length_penalty(l_a: int, l_r: int) -> float:
    """Piecewise L(q) from the metric. Lengths are token counts.

    L = 1                      if l_a <= 1.5*l_r   (free length)
    L = -(2/3)*(l_a/l_r) + 2   if 1.5*l_r < l_a < 3*l_r
    L = 0                      if l_a >= 3*l_r
    """
    if l_r <= 0:
        # Degenerate reference: only an (also-empty) answer escapes the penalty.
        return 1.0 if l_a == 0 else 0.0
    ratio = l_a / l_r
    if ratio <= 1.5:
        return 1.0
    if ratio >= 3.0:
        return 0.0
    return -(2.0 / 3.0) * ratio + 2.0


@dataclass
class RecallLResult:
    mean_recall_l: float
    mean_bert_recall: float
    mean_length_penalty: float
    n: int
    per_question: pl.DataFrame  # q_id, bert_recall, l_a, l_r, length_penalty, recall_l

    def summary(self) -> str:
        zeroed = int((self.per_question["length_penalty"] == 0.0).sum())
        penalized = int((self.per_question["length_penalty"] < 1.0).sum())
        return (
            f"Recall-L = {self.mean_recall_l:.4f}  "
            f"(R_BERT={self.mean_bert_recall:.4f}  L={self.mean_length_penalty:.4f})  "
            f"n={self.n}\n"
            f"  length-penalized (L<1): {penalized}/{self.n} "
            f"({penalized / self.n:.1%}); zeroed (L=0): {zeroed}/{self.n} "
            f"({zeroed / self.n:.1%})"
        )


def _count_tokens(texts: list[str], tokenizer_name: str) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    return [len(ids) for ids in enc]


def recall_l(
    q_ids: list[str],
    predictions: list[str],
    references: list[str],
    *,
    lang: str = "ru",
    model_type: str | None = None,
    num_layers: int | None = None,
    length_tokenizer: str = "bert-base-multilingual-cased",
    batch_size: int = 64,
    device: str | None = None,
    idf: bool = False,
) -> RecallLResult:
    """Score predictions against references with BERTScore-recall × length penalty.

    `model_type`/`num_layers` override the BERTScore backbone; leave `None` to let
    `lang` pick it (`ru` → multilingual BERT). `length_tokenizer` counts l_a/l_r.
    """
    if not (len(q_ids) == len(predictions) == len(references)):
        raise ValueError("q_ids, predictions and references must be the same length")
    if not q_ids:
        raise ValueError("nothing to score (empty input)")

    from bert_score import score as bertscore

    # bert-score's sent_encode hits a broken empty-string branch for blank/whitespace
    # input; substitute a real non-whitespace placeholder so it takes the normal path.
    # Length below is still counted on the RAW text, so a truly empty answer stays 0 tok.
    empty_placeholder = "."
    cands = [p if p.strip() else empty_placeholder for p in predictions]
    refs = [r if r.strip() else empty_placeholder for r in references]

    _, recall, _ = bertscore(
        cands,
        refs,
        lang=lang,
        model_type=model_type,
        num_layers=num_layers,
        idf=idf,
        batch_size=batch_size,
        device=device,
        rescale_with_baseline=False,
        verbose=False,
    )
    bert_recall = [float(x) for x in recall]

    # Length is measured on the *raw* prediction/reference (empty → 0 tokens).
    l_a = _count_tokens([p if p.strip() else "" for p in predictions], length_tokenizer)
    l_r = _count_tokens([r if r.strip() else "" for r in references], length_tokenizer)
    penalties = [length_penalty(a, r) for a, r in zip(l_a, l_r)]
    scores = [br * pen for br, pen in zip(bert_recall, penalties)]

    per_q = pl.DataFrame(
        {
            "q_id": q_ids,
            "bert_recall": bert_recall,
            "l_a": l_a,
            "l_r": l_r,
            "length_penalty": penalties,
            "recall_l": scores,
        }
    )
    n = len(q_ids)
    return RecallLResult(
        mean_recall_l=sum(scores) / n,
        mean_bert_recall=sum(bert_recall) / n,
        mean_length_penalty=sum(penalties) / n,
        n=n,
        per_question=per_q,
    )
