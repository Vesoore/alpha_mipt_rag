"""Profile the raw RAG corpus and queries.

Outputs go to stdout. Re-run any time the data changes. No pipeline code lives
here — this is a data-inspection step that informs chunking + retrieval design.
"""

import re
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEBSITES = DATA_DIR / "websites.csv"
QUESTIONS = DATA_DIR / "questions.csv"

CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
LATIN_RE = re.compile(r"[A-Za-z]")
HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]{0,200}>")
WHITESPACE_RUN_RE = re.compile(r"\s{3,}")
NAV_HINT_RE = re.compile(
    r"(?i)(куки|cookie|подписаться|copyright|©|меню|навигац|footer|header)"
)


def hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def percentiles(s: pl.Series, qs=(0.5, 0.75, 0.9, 0.95, 0.99, 1.0)) -> dict[str, float]:
    out = {}
    for q in qs:
        v = s.quantile(q)
        out[f"p{int(q * 100)}"] = float(v) if v is not None else float("nan")
    return out


def profile_websites() -> pl.DataFrame:
    df = pl.read_csv(WEBSITES, infer_schema_length=0)  # keep everything as utf8

    hr("websites.csv — shape & schema")
    print(f"rows: {df.height:,}")
    print(f"cols: {df.columns}")
    print(df.schema)

    hr("null / empty counts per column")
    for col in df.columns:
        n_null = df[col].null_count()
        n_empty = df.filter(pl.col(col).fill_null("").str.len_chars() == 0).height
        print(f"  {col:<8}  null={n_null:<6} empty={n_empty}")

    hr("kind distribution")
    print(df.group_by("kind").len().sort("len", descending=True))

    hr("duplicates")
    dup_url = df.height - df.select(pl.col("url").n_unique()).item()
    dup_text = df.height - df.select(pl.col("text").n_unique()).item()
    dup_title = df.height - df.select(pl.col("title").n_unique()).item()
    print(f"  duplicate urls (count - n_unique):   {dup_url}")
    print(f"  duplicate texts:                     {dup_text}")
    print(f"  duplicate titles:                    {dup_title}")

    hr("text length (chars)")
    # work on non-null text only
    chars = df.with_columns(
        pl.col("text").fill_null("").str.len_chars().alias("n_chars"),
        pl.col("text").fill_null("").str.split(" ").list.len().alias("n_words_approx"),
    )
    n_chars = chars["n_chars"]
    n_words = chars["n_words_approx"]
    print(f"  chars: min={n_chars.min()} max={n_chars.max()} mean={n_chars.mean():.0f}")
    print(f"  chars percentiles: {percentiles(n_chars)}")
    print(
        f"  words(approx): min={n_words.min()} max={n_words.max()} mean={n_words.mean():.0f}"
    )
    print(f"  words percentiles: {percentiles(n_words)}")

    # very-short and very-long text
    short = chars.filter(pl.col("n_chars") < 100).height
    long_ = chars.filter(pl.col("n_chars") > 20_000).height
    print(f"  docs < 100 chars:    {short}")
    print(f"  docs > 20k chars:    {long_}")

    hr("language heuristic (cyrillic share of letters)")
    # sample to avoid scanning 23MB row-by-row with python regex when not needed
    sample = df.sample(n=min(1000, df.height), seed=42, with_replacement=False)
    cyr_share = []
    for t in sample["text"].to_list():
        if not t:
            continue
        cyr = len(CYRILLIC_RE.findall(t))
        lat = len(LATIN_RE.findall(t))
        total = cyr + lat
        if total == 0:
            continue
        cyr_share.append(cyr / total)
    s = pl.Series("cyr_share", cyr_share)
    print(f"  sample size: {len(cyr_share)}")
    print(
        f"  cyrillic share: mean={s.mean():.3f} p10={s.quantile(0.1):.3f} "
        f"p50={s.quantile(0.5):.3f} p90={s.quantile(0.9):.3f}"
    )
    mostly_latin = sum(1 for x in cyr_share if x < 0.3)
    print(f"  docs in sample with <30% cyrillic: {mostly_latin}")

    hr("HTML / nav-junk hints (sample of 1000)")
    n_html = 0
    n_wsrun = 0
    n_nav = 0
    for t in sample["text"].to_list():
        if not t:
            continue
        if HTML_TAG_RE.search(t):
            n_html += 1
        if WHITESPACE_RUN_RE.search(t):
            n_wsrun += 1
        if NAV_HINT_RE.search(t):
            n_nav += 1
    print(f"  docs with html-ish tags:        {n_html} / {len(sample)}")
    print(f"  docs with 3+ whitespace run:    {n_wsrun} / {len(sample)}")
    print(f"  docs hitting nav/cookie hints:  {n_nav} / {len(sample)}")

    hr("url host distribution (top 10)")
    hosts = (
        df.with_columns(
            pl.col("url")
            .fill_null("")
            .str.extract(r"https?://([^/]+)", 1)
            .alias("host")
        )
        .group_by("host")
        .len()
        .sort("len", descending=True)
        .head(10)
    )
    print(hosts)

    hr("examples — first non-null text per kind")
    for k in df["kind"].unique().to_list():
        if k is None:
            continue
        row = df.filter((pl.col("kind") == k) & pl.col("text").is_not_null()).head(1)
        if row.height == 0:
            continue
        t = row["text"].item()
        title = row["title"].item()
        url = row["url"].item()
        print(f"\n--- kind={k!r}  title={title!r}\n    url={url}")
        print(f"    text[:400]: {t[:400]!r}")

    return df


def profile_questions() -> pl.DataFrame:
    df = pl.read_csv(QUESTIONS, infer_schema_length=0)

    hr("questions.csv — shape & schema")
    print(f"rows: {df.height:,}")
    print(f"cols: {df.columns}")
    print(df.schema)

    hr("null / empty / duplicate counts")
    for col in df.columns:
        n_null = df[col].null_count()
        n_empty = df.filter(pl.col(col).fill_null("").str.len_chars() == 0).height
        print(f"  {col:<8}  null={n_null:<6} empty={n_empty}")
    dup_q = df.height - df.select(pl.col("query").n_unique()).item()
    dup_id = df.height - df.select(pl.col("q_id").n_unique()).item()
    print(f"  duplicate queries: {dup_q}")
    print(f"  duplicate q_ids:   {dup_id}")

    hr("query length (chars / words)")
    qlen = df.with_columns(
        pl.col("query").fill_null("").str.len_chars().alias("n_chars"),
        pl.col("query").fill_null("").str.split(" ").list.len().alias("n_words"),
    )
    nc = qlen["n_chars"]
    nw = qlen["n_words"]
    print(f"  chars: min={nc.min()} max={nc.max()} mean={nc.mean():.1f}")
    print(f"  chars percentiles: {percentiles(nc)}")
    print(f"  words: min={nw.min()} max={nw.max()} mean={nw.mean():.1f}")
    print(f"  words percentiles: {percentiles(nw)}")

    hr("language heuristic")
    cyr_share = []
    for q in df["query"].to_list():
        if not q:
            continue
        cyr = len(CYRILLIC_RE.findall(q))
        lat = len(LATIN_RE.findall(q))
        total = cyr + lat
        if total == 0:
            continue
        cyr_share.append(cyr / total)
    s = pl.Series("cyr_share", cyr_share)
    print(
        f"  cyrillic share: mean={s.mean():.3f} p10={s.quantile(0.1):.3f} "
        f"p50={s.quantile(0.5):.3f} p90={s.quantile(0.9):.3f}"
    )
    print(f"  queries with <30% cyrillic: {sum(1 for x in cyr_share if x < 0.3)}")

    hr("examples — first 10 queries")
    for row in df.head(10).iter_rows(named=True):
        print(f"  {row['q_id']}  {row['query']!r}")

    return df


def main() -> None:
    print("DATA_DIR =", DATA_DIR)
    profile_websites()
    profile_questions()


if __name__ == "__main__":
    main()
