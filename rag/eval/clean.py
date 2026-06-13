"""Strip RAG-generation artifacts from reference/prediction text before scoring.

`data/sample_submission.csv` is itself the output of some baseline RAG, so its
"reference" answers carry meta-references the true answer would never contain —
"Согласно Фрагменту 2, …", "В предоставленном контексте указано: …". Left in, they
distort BERTScore-recall and length. We remove the meta-scaffolding and keep the
factual content. Conservative by design: if cleaning would gut the string, keep the
original.
"""

from __future__ import annotations

import re

# A "fragment/context/source" referent the model cites instead of stating the fact.
_REFERENT = r"(?:фрагмент\w*\s*№?\s*\d+|(?:предоставленн\w+\s+)?контекст\w*|(?:предоставленн\w+\s+)?информаци\w+|источник\w*\s*№?\s*\d+)"

# Plural / numberless fragment referent, only safe inside a comma/colon-bounded clause.
_REFERENT_LOOSE = rf"(?:{_REFERENT}|(?:предоставленн\w+\s+)?фрагмент\w*)"

# Meta-clauses: a citation lead-in ending at a comma/colon — drop the whole clause.
_CLAUSE_PATTERNS = [
    rf"(?i)\bсогласно\s+{_REFERENT_LOOSE}\s*[,:]\s*",
    rf"(?i)\bв\s+соответствии\s+с\s+{_REFERENT_LOOSE}\s*[,:]\s*",
    rf"(?i)\b(?:на основании|на основе|исходя\s+из)\s+{_REFERENT_LOOSE}\s*[,:]\s*",
    rf"(?i)\bв\s+{_REFERENT}\s+(?:указано|сказано|говорится|написано|описано|сообщается)\s*[,:]\s*",
    rf"(?i)\bкак\s+(?:указано|сказано|описано)\s+(?:в|во)\s+{_REFERENT}\s*[,:]?\s*",
]

# Bare in-sentence mentions left without a clause delimiter.
_INLINE_PATTERNS = [
    rf"(?i)\b(?:во?|из|согласно)\s+{_REFERENT}\b",
    r"(?i)\bв\s+предоставленн\w+\s+контекст\w*\b",
    r"(?i)\bфрагмент\w*\s*№?\s*\d+\b",
]

_NO_DATA_RE = re.compile(r"(?i)^\s*нет\s+ответа\b")


def is_no_data_answer(text: str) -> bool:
    """True if an answer is an abstention ('Нет ответа', with/without trailing dot)."""
    return bool(_NO_DATA_RE.match(text or ""))


_CLAUSE_RE = [re.compile(p) for p in _CLAUSE_PATTERNS]
_INLINE_RE = [re.compile(p) for p in _INLINE_PATTERNS]
_LEAD_JUNK_RE = re.compile(r"^[\s,:;.—\-«\"']+")
_HSPACE_RE = re.compile(r"[ \t]+")


def strip_rag_artifacts(text: str) -> str:
    if not text or not text.strip():
        return text
    out = text
    for r in _CLAUSE_RE:
        out = r.sub(" ", out)
    for r in _INLINE_RE:
        out = r.sub(" ", out)
    out = _HSPACE_RE.sub(" ", out)
    # tidy spaces left in front of punctuation
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = _LEAD_JUNK_RE.sub("", out).strip()
    if out:
        out = out[0].upper() + out[1:]
    # If cleaning destroyed the answer (e.g. it was *only* a citation), keep the original.
    if len(out) < max(5, int(0.3 * len(text.strip()))):
        return text.strip()
    return out
