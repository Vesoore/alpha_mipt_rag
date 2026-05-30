"""Quick look at extreme-length docs that will shape chunking + cleaning rules."""

from pathlib import Path

import polars as pl

WEBSITES = Path(__file__).resolve().parent.parent / "data" / "websites.csv"


def main() -> None:
    df = pl.read_csv(WEBSITES, infer_schema_length=0).with_columns(
        pl.col("text").str.len_chars().alias("n_chars")
    )

    print("=== top 5 longest docs ===")
    for r in df.sort("n_chars", descending=True).head(5).iter_rows(named=True):
        print(
            f"  web_id={r['web_id']}  kind={r['kind']}  n_chars={r['n_chars']:,}\n"
            f"    title={r['title'][:120]!r}\n"
            f"    url={r['url']}\n"
            f"    text[:300]={r['text'][:300]!r}\n"
        )

    print("=== 5 shortest non-trivial docs ===")
    for r in df.sort("n_chars").head(5).iter_rows(named=True):
        print(
            f"  web_id={r['web_id']}  kind={r['kind']}  n_chars={r['n_chars']}\n"
            f"    title={r['title'][:120]!r}\n"
            f"    url={r['url']}\n"
            f"    text={r['text']!r}\n"
        )

    print("=== duplicate-text examples ===")
    dups = (
        df.group_by("text")
        .agg(pl.col("web_id"), pl.col("url"), pl.col("title"))
        .filter(pl.col("web_id").list.len() > 1)
        .head(5)
    )
    for r in dups.iter_rows(named=True):
        print(
            f"  web_ids={r['web_id']}  n_chars={len(r['text'])}\n"
            f"    titles={r['title']}\n"
            f"    urls={r['url']}\n"
        )

    print("=== alfabank.by + auth + job hosts ===")
    weird = df.with_columns(
        pl.col("url").str.extract(r"https?://([^/]+)", 1).alias("host")
    ).filter(
        pl.col("host").is_in(
            ["www.alfabank.by", "private.auth.alfabank.ru", "job.alfabank.ru", "alfabank.st"]
        )
    )
    for r in weird.iter_rows(named=True):
        print(f"  host={r['url'][:80]}  title={r['title'][:80]!r}  n_chars={r['n_chars']}")


if __name__ == "__main__":
    main()
