"""Minimal Ollama chat client for docs QA eval. No backend dependencies; stdlib only."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"


def get_chat_completion(
    *,
    model: str | None = None,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    timeout: int = 60,
    max_output_tokens: int = 1024,
    response_format: dict[str, Any] | None = None,
    think: bool | None = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Call Ollama /api/chat. Returns the assistant message content (or JSON string if response_format set)."""
    base_url = (os.environ.get("OLLAMA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL

    options: dict[str, Any] = {
        "temperature": temperature,
        "num_predict": max_output_tokens,
    }
    if think is not None:
        options["think"] = think

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    if response_format is not None:
        schema = response_format.get("json_schema", response_format)
        if isinstance(schema, dict) and "schema" in schema:
            schema = schema["schema"]
        body["format"] = schema if isinstance(schema, dict) else {"type": "json_object"}

    req = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            message = data.get("message")
            if not message:
                raise RuntimeError(f"Ollama response missing 'message': {data}")
            return message.get("content", "").strip()

        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            err = RuntimeError(
                f"Ollama chat HTTP {e.code} (model '{model}'). "
                f"Response: {detail or e.reason}"
            )
            if e.code < 500 or attempt == attempts - 1:
                raise err from e

        except urllib.error.URLError as e:
            err = RuntimeError(
                f"Ollama request failed (is Ollama running at {base_url}?). Error: {e}"
            )
            if attempt == attempts - 1:
                raise err from e

        except RuntimeError:
            if attempt == attempts - 1:
                raise

        time.sleep(retry_delay * (2 ** attempt))

    raise RuntimeError("Ollama request failed after retries")  # unreachable, satisfies type checker
