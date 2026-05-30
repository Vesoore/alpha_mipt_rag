"""Module contract types — every retrieval/grounding/generation function speaks these."""

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    web_id: str
    url: str
    text: str
    score: float
    title: str = ""


class GroundingContext(BaseModel):
    model_config = {"frozen": True}

    q_id: str
    query: str
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    context_str: str


class Answer(BaseModel):
    model_config = {"frozen": True}

    q_id: str
    answer: str
