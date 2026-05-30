"""Post-trim generated answers to fit the Recall-L length penalty.

The metric kills any answer >=3x reference length and grants free length up to
1.5x. Without per-question reference lengths we use a static character budget
(tuned by spot-check). Trim at the last sentence boundary when possible.
"""

SENTENCE_ENDS = ".!?…"


def trim_answer(answer: str, max_chars: int) -> str:
    a = answer.strip()
    if len(a) <= max_chars:
        return a
    head = a[:max_chars]
    last_end = max(head.rfind(c) for c in SENTENCE_ENDS)
    if last_end >= int(max_chars * 0.4):
        return head[: last_end + 1].strip()
    last_space = head.rfind(" ")
    if last_space > 0:
        return head[:last_space].rstrip() + "…"
    return head.rstrip()
