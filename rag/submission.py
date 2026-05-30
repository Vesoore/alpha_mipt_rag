"""Write submission CSV conforming to data/sample_submission.csv."""

from pathlib import Path

import polars as pl

from rag.types import Answer


def write_submission(
    answers: list[Answer],
    sample_path: str | Path,
    out_path: str | Path,
) -> None:
    sample_cols = pl.read_csv(sample_path, n_rows=0).columns

    if sample_cols == ["q_id", "answer"]:
        text_col = "answer"
    elif sample_cols == ["q_id", "answer_new"]:
        text_col = "answer_new"
    else:
        raise ValueError(
            f"Unexpected sample_submission columns: {sample_cols}. "
            f"Expected ['q_id', 'answer'] or ['q_id', 'answer_new']."
        )

    df = pl.DataFrame(
        {
            "q_id": [a.q_id for a in answers],
            text_col: [a.answer for a in answers],
        }
    ).select(sample_cols)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out_path)
    print(f"[submission] wrote {df.height} rows → {out_path} " f"(cols={sample_cols})")
