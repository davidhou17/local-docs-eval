"""Responder registry for docs Q&A evaluation."""

from .base import Responder
from .golden_answer import GoldenAnswerResponder
from .naive_baseline import NaiveBaselineResponder
from .rag_responder import RagResponder

REGISTRY: dict[str, type] = {
    "golden_answer": GoldenAnswerResponder,
    "naive_baseline": NaiveBaselineResponder,
    "rag": RagResponder,
}


def get_responder(name: str, **kwargs) -> Responder:
    """Return a responder instance by name. kwargs are passed to the responder constructor."""
    if name not in REGISTRY:
        raise ValueError(f"Unknown responder: {name}. Choices: {list(REGISTRY.keys())}")
    cls = REGISTRY[name]
    return cls(**kwargs)
