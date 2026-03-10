"""Responder protocol for docs Q&A evaluation."""

from typing import Protocol


class Responder(Protocol):
    """Protocol for responders that produce an answer given a question and context."""

    def get_response(
        self,
        question_id: str,
        question: str,
        category: str,
        golden_answer: str,
    ) -> str:
        """Return the model response for the given question."""
        ...


def get_response(
    question_id: str,
    question: str,
    category: str,
    golden_answer: str,
    responder: Responder,
) -> str:
    """Call the responder's get_response."""
    return responder.get_response(
        question_id=question_id,
        question=question,
        category=category,
        golden_answer=golden_answer,
    )
