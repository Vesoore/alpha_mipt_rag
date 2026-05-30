"""Load config.yaml into a typed model and seed every RNG used by the pipeline."""

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class Paths(BaseModel):
    websites_csv: str
    questions_csv: str
    sample_submission_csv: str
    artifacts_dir: str
    chunks_parquet: str
    faiss_index: str
    bm25_pickle: str
    submission_csv: str
    models_dir: str


class Cleaning(BaseModel):
    min_chars: int
    dedup_prefix_chars: int
    drop_hosts: list[str]


class Chunking(BaseModel):
    target_tokens: int
    overlap_tokens: int
    tokenizer_model: str


class Embedder(BaseModel):
    model: str
    device: str
    batch_size: int
    query_prefix: str
    passage_prefix: str
    normalize: bool


class Retrieval(BaseModel):
    dense_top_n: int
    bm25_top_n: int
    rrf_k: int
    fused_top_n: int


class Reranker(BaseModel):
    model: str
    device: str
    batch_size: int
    top_k: int


class Grounding(BaseModel):
    context_max_tokens: int
    near_dup_prefix_chars: int


class Generator(BaseModel):
    backend: str = "vllm"
    model: str = ""
    model_path: str | None = None  # llama_cpp only
    quantization: str | None = None  # vllm only
    tensor_parallel_size: int = 1  # vllm only
    gpu_memory_utilization: float = 0.85  # vllm only
    max_model_len: int | None = None  # vllm only; cap context to shrink KV cache
    enforce_eager: bool = False  # vllm only; skip CUDA graph capture (more robust init)
    n_ctx: int | None = None  # llama_cpp only
    n_gpu_layers: int | None = None  # llama_cpp only
    temperature: float
    top_p: float
    max_new_tokens: int


class Length(BaseModel):
    answer_max_chars: int
    no_data_phrase: str


class Config(BaseModel):
    seed: int
    paths: Paths
    cleaning: Cleaning
    chunking: Chunking
    embedder: Embedder
    retrieval: Retrieval
    reranker: Reranker
    grounding: Grounding
    generator: Generator
    length: Length

    def resolve(self, rel: str) -> Path:
        """Resolve a config-relative path against the repo root."""
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
