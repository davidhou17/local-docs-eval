"""LLM-as-judge: compare model response to expected answer, output 1 / 0.75 / 0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from .llm_ollama import get_chat_completion

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "docs_qa_judge_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "enum": [1.0, 0.75, 0.0],
                },
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
            "additionalProperties": False,
        },
    },
}

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for documentation Q&A. Compare the model's response to the expected (golden) answer and assign a score.

Scoring:
- 1.0 (Correct): The response is accurate and complete relative to the expected answer (or equivalent).
- 0.75 (Partial): A non-trivial subset of the expected answer is present and correct; key information is included but may be incomplete or slightly off.
- 0.0 (Wrong): The response is incorrect, contradictory, irrelevant, or missing the key information from the expected answer.

Output only valid JSON with "score" (1.0, 0.75, or 0.0) and "reasoning" (one or two sentences)."""


@dataclass
class JudgeResult:
    score: Literal[1.0, 0.75, 0.0]
    reasoning: str


def judge(
    question: str,
    expected_answer: str,
    model_response: str,
    model: str = "qwen3:8b",
    timeout: int = 120,
    max_retries: int = 3,
) -> JudgeResult:
    """Run LLM-as-judge: compare model_response to expected_answer; return score and reasoning."""
    user_content = f"""## Question
{question}

## Expected answer
{expected_answer}

## Model response
{model_response}

## Task
Compare the model response to the expected answer and assign a score (1.0, 0.75, or 0.0) with brief reasoning."""

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    response = get_chat_completion(
        model=model,
        messages=messages,
        temperature=0.0,
        timeout=timeout,
        max_output_tokens=4096,
        response_format=JUDGE_RESPONSE_FORMAT,
        think=False,
        max_retries=max_retries,
    )

    if not response:
        raise RuntimeError(
            f"Judge LLM returned empty response (model={model!r}). "
            "The model may have exhausted its token budget on internal reasoning."
        )

    try:
        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Judge LLM returned invalid JSON (model={model!r}): {response[:200]!r}"
        ) from exc

    score = float(result["score"])
    if score not in (1.0, 0.75, 0.0):
        score = 0.0
    return JudgeResult(score=score, reasoning=result.get("reasoning", "") or "")
