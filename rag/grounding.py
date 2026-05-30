"""Assemble retrieved chunks into a context block under a token budget."""

from rag.config import Grounding as GroundingCfg
from rag.types import GroundingContext, RetrievedChunk


def _dedup(chunks: list[RetrievedChunk], prefix_chars: int) -> list[RetrievedChunk]:
    seen: set[int] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        h = hash(c.text[:prefix_chars])
        if h in seen:
            continue
        seen.add(h)
        out.append(c)
    return out


def assemble_context(
    q_id: str,
    query: str,
    chunks: list[RetrievedChunk],
    cfg: GroundingCfg,
    tokenizer,
) -> GroundingContext:
    deduped = _dedup(chunks, cfg.near_dup_prefix_chars)

    parts: list[str] = []
    used: list[RetrievedChunk] = []
    total = 0
    for i, c in enumerate(deduped, start=1):
        block = f"[источник {i}: {c.url}]\n{c.text}\n\n"
        n = len(tokenizer.encode(block, add_special_tokens=False))
        if used and total + n > cfg.context_max_tokens:
            break
        parts.append(block)
        used.append(c)
        total += n

    return GroundingContext(
        q_id=q_id,
        query=query,
        chunks=used,
        context_str="".join(parts).rstrip(),
    )
