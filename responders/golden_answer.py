"""Sanity-check responder: returns the golden answer as-is (expect ~1.0 from judge)."""


class GoldenAnswerResponder:
    """Responder that returns the expected answer. Use to validate the pipeline."""

    def __init__(self, **_kwargs: object):
        pass  # no LLM calls; absorb any shared kwargs (model, timeout, etc.)

    def get_response(
        self,
        question_id: str,
        question: str,
        category: str,
        golden_answer: str,
    ) -> str:
        return golden_answer
