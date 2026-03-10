"""Naive baseline: LLM with question only, no doc context (baseline to beat)."""

from ..llm_ollama import get_chat_completion

SYSTEM_PROMPT = """You are a product expert. Answer the following question based on your knowledge. If you don't know, say so. Be concise."""


class NaiveBaselineResponder:
    """Responder that calls the LLM with only the question (no RAG)."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        timeout: int = 120,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
        think: bool | None = None,
        **_kwargs: object,
    ):
        self.model = model
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.think = think

    def get_response(
        self,
        question_id: str,
        question: str,
        category: str,
        golden_answer: str,
    ) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        response = get_chat_completion(
            model=self.model,
            messages=messages,
            temperature=0.0,
            timeout=self.timeout,
            max_output_tokens=self.max_output_tokens,
            think=self.think,
            max_retries=self.max_retries,
        )
        return response.strip() if response else ""
