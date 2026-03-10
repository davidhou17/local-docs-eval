"""RAG responder: retrieve relevant doc chunks, then ask the LLM to answer grounded in context."""

from ..llm_ollama import get_chat_completion
from ..rag import build_index, retrieve, DEFAULT_EMBED_MODEL

SYSTEM_PROMPT = """You are a helpful product documentation assistant. Answer the question using ONLY the provided documentation context. If the context doesn't contain enough information, say so. Be concise and accurate."""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        title = chunk.get("title", "")
        header = f"[{i}] {title} ({source})" if title else f"[{i}] {source}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


class RagResponder:
    """Responder that retrieves doc chunks via embedding similarity, then generates an answer."""

    def __init__(
        self,
        model: str = "qwen3:8b",
        docs_dir: str = "",
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = 5,
        timeout: int = 120,
        max_output_tokens: int = 1024,
        max_retries: int = 3,
        think: bool | None = None,
        **_kwargs: object,
    ):
        self.model = model
        self.embed_model = embed_model
        self.top_k = top_k
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.think = think
        if not docs_dir:
            raise ValueError("docs_dir is required for the rag responder")
        self.index = build_index(docs_dir, embed_model=embed_model)

    def get_response(
        self,
        question_id: str,
        question: str,
        category: str,
        golden_answer: str,
    ) -> str:
        chunks = retrieve(question, self.index, embed_model=self.embed_model, top_k=self.top_k)
        context = _format_context(chunks)

        user_content = f"""## Documentation context
{context}

## Question
{question}

Answer the question based on the documentation context above."""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
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
