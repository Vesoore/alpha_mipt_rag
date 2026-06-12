# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG system over Alfa-Bank web content ("alpha-mipt-rag"), built for the Alfa-Bank ×
MIPT hackathon. Given a user question, the pipeline retrieves relevant fragments from
a fixed corpus of scraped `alfabank.ru` pages and generates a grounded answer with a
**local** LLM.

Current state: skeleton. `main.py` is a placeholder hello-world, `notebook.ipynb` is
empty. You are building this up, not navigating an existing system. The target
end-to-end flow is **retrieval → grounding → generation**.

The scored metric is **Recall-L** (BERTScore-recall × a length penalty). Every design
decision serves it: coverage of the reference answer matters most, but over-long
answers are penalized and answers ≥3× the reference score zero. See "Metric".

## Hard constraints (violation = disqualification)

- **Open source only.** All models/libraries must have a free, permissive license.
  Verify the license before adopting any model.
- **No external/closed generation APIs.** Inference runs **locally** (no OpenAI,
  Anthropic, Gemini, YandexGPT, GigaChat, etc.). No network calls on the inference
  hot path — downloading weights from HF is an environment-setup step only.
- **Corpus is fixed.** No extra/closed data, no adding external data to the index.
- Answers must not contain confidential info or knowingly incorrect/dangerous advice.
- **Submission limit: 3 CSV uploads per day** (00:00–23:59 MSK). Always run the
  offline checklist (see "Metric") before spending one. Do not burn submissions blind.
- Top-10 solutions go through code review → keep code clean, reproducible, seeded,
  with no hardcoded paths.

## Tooling

Managed with `uv` (see `uv.lock`, `pyproject.toml`, `.python-version` pinned to 3.13).

- Install / sync deps: `uv sync`
- Run the entrypoint: `uv run python main.py`
- Add a dependency: `uv add <pkg>` (do **not** hand-edit `pyproject.toml` + run pip)
- Open the notebook: `uv run jupyter lab notebook.ipynb` (jupyter is not yet a declared
  dep — `uv add --dev jupyterlab` first)

`polars` is the only declared runtime dep right now — prefer it over pandas for any
tabular work on the CSVs below.

Inference backends to add when chosen (pick per available VRAM — see "Models"):
GPU → `vllm` or `transformers` + quantization (bitsandbytes/AWQ/GPTQ); CPU/low-VRAM →
`llama-cpp-python` + GGUF. Retrieval stack: an embedder (sentence-transformers), an ANN
index (`faiss-cpu`/`faiss-gpu`), and BM25 (`rank_bm25` or `bm25s`).

## Data

`data/` holds the corpus the RAG is built over. Not loaded by any code yet — wire-up is
part of the work. Assume UTF-8 and that `text` cells contain embedded newlines.

- `data/websites.csv` (~23 MB): scraped Alfa-Bank pages. Columns:
  `web_id, url, kind, title, text`. `text` is long-form Russian prose — the retrieval
  corpus. (`url` is the source link; keep it for traceability of facts.)
- `data/questions.csv` (~750 KB): evaluation queries. Columns: `q_id, query`.
  Russian-language questions a user might ask of the corpus.

Output: a submission CSV whose columns match `sample_submission.csv` exactly (verify
column names/order/encoding programmatically before writing).

**Inspect the real data before writing code.** Profile `text` lengths, detect
HTML/navigation junk and duplicate pages, check per-document language, and find empty
rows. Do not design chunking blind.

## Architecture

```
query
 ├─ [0] query normalization (clean; optional expansion)
 ├─ [1] RETRIEVAL (hybrid)
 │       dense bi-encoder + FAISS  ⊕  BM25  → RRF fusion → top-N (~50–100)
 ├─ [2] RERANK (cross-encoder)     top-N → top-k (~3–8, tune it)
 ├─ [3] GROUNDING  assemble context from top-k, dedupe, trim to token budget
 └─ [4] GENERATION local instruct LLM
         prompt: "answer only from the context; if absent, say there is no data"
         post-process length to fit Recall-L; optional self-check
```

Hybrid retrieval is deliberate: dense catches paraphrase, BM25 catches exact terms
(card/tariff names) that dense often misses.

### Chunking
Split on structure (headings/paragraphs), not raw characters. Start ~512 tokens with
~64–128 overlap and tune by Recall-L. Keep a `chunk_id → web_id → url` mapping so every
answer is traceable to its source (this is the value of RAG and helps debug
hallucinations).

### Models
Corpus is Russian (questions too); use **multilingual** models so RU + any EN terms are
both handled. Hardware is not yet known — keep model names in `config.yaml` (never
hardcode) and pick a profile by actual VRAM:

| Profile     | Embedder                                          | Reranker                  | Generator                                  |
|-------------|---------------------------------------------------|---------------------------|--------------------------------------------|
| GPU 24GB+   | `intfloat/multilingual-e5-large` / `BAAI/bge-m3`  | `BAAI/bge-reranker-v2-m3` | 7–9B instruct (e.g. Qwen2.5-7B-Instruct)   |
| GPU 8–16GB  | `multilingual-e5-base` / `bge-m3`                 | `bge-reranker-base`       | 3–4B instruct + quantization               |
| CPU only    | `multilingual-e5-small`                           | optional / slow           | 1.5–3B quantized (GGUF, llama.cpp)         |

Low temperature (~0–0.3) for factuality; fixed `seed`; `max_new_tokens` set against the
target length (see "Metric").

## Module contract

Keep one explicit format at module boundaries (use dataclasses/pydantic):

```python
RetrievedChunk   = {"chunk_id": str, "web_id": str, "url": str, "text": str, "score": float}
GroundingContext = {"q_id": str, "query": str, "chunks": list[RetrievedChunk], "context_str": str}
Answer           = {"q_id": str, "answer": str}
```

## Metric — Recall-L

`Recall_L(q) = R_BERT(q) · L(q)`, averaged over all questions.
`R_BERT(q)` is the recall component of BERTScore (semantic coverage, not exact words).
With `l_a` = answer length, `l_r` = reference length (tokens):

```
L(q) = 1                     if  l_a ≤ 1.5·l_r
L(q) = -(2/3)·(l_a/l_r) + 2  if  1.5·l_r < l_a < 3·l_r
L(q) = 0                     if  l_a ≥ 3·l_r
```

Implications for generation:
- Length up to **1.5×** the reference is free — use it to maximize coverage.
- Cover as many relevant facts from the context as possible (raises recall).
- Never let `l_a ≥ 3·l_r` — it zeroes the question. Cap `max_new_tokens` and post-trim
  with margin.
- Balance more facts (↑ recall) vs. length (↓ L); tune top-k and answer length together.

### Pre-submission checklist (spends 1 of 3 daily uploads)
- [ ] Local Recall-L scorer implemented (multilingual BERTScore-recall + piecewise L(q)).
- [ ] Local hold-out evaluated, **or** manual spot-check of a sample for gross errors.
- [ ] Full `questions.csv` run with no crashes and no empty answers.
- [ ] `l_a/l_r` distribution checked; tail ≥3× trimmed.
- [ ] "No relevant data" cases handled (does not hallucinate).
- [ ] CSV matches `sample_submission.csv` exactly (columns, UTF-8, delimiter, quoting).
- [ ] Seeds, model versions, and `config.yaml` for this submission logged.
- [ ] No network calls on the inference path (weights are local).

## Working rules

- Data first, code second — validate chunking/retrieval hypotheses on the real CSVs.
- Change one component at a time and measure the Recall-L delta offline; log every
  experiment (config → score).
- Don't spend submissions blind — only after the checklist passes.
- Everything configurable lives in `config.yaml`; no magic numbers or paths in code.
- Fix all seeds (`random`, `numpy`, `torch`). Reproducibility is a code-review condition.
- Check each model's license and that inference is local before using it.
- Don't commit data or model weights to git.
- When a constraint or the metric is ambiguous, ask — don't guess.

## Definition of Done

1. Answers are meaningful and relevant to the query.
2. Answers are grounded in retrieved fragments (minimal hallucination; traceable to `url`).
3. Generator integrated with retrieval via the shared module contract.
4. "No relevant information" handled correctly.
5. Submission CSV in `sample_submission.csv` format passes the platform.
6. Pipeline reproducible from a clean env via `uv sync` + `config.yaml`.
