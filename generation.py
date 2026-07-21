import os
from openai import OpenAI

_MODEL = "llama-3.3-70b-versatile"   
_BASE_URL = "https://api.groq.com/openai/v1"
_REFUSAL_SENTINEL = "NOT_IN_CONTEXT"

_SYSTEM_PROMPT = """You answer questions about legal documents.

Rules:
- Use ONLY the numbered context passages provided. Never use outside knowledge.
- Cite the passage number for every claim, e.g. [2].
- If the passages do not answer the question, reply with exactly: NOT_IN_CONTEXT
- Quote contract language verbatim where the exact wording matters.
- Be concise."""

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at console.groq.com."
            )
        _client = OpenAI(base_url=_BASE_URL, api_key=key)
    return _client


def build_context(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({r['meta']['doc_name']}, page {r['meta']['page_number']})\n{r['meta']['text']}"
        for i, r in enumerate(results, start=1)
    )


def generate_answer(question: str, results: list[dict]) -> str | None:
    """Answer strictly from retrieved passages. Returns None if context is insufficient."""
    response = _get_client().chat.completions.create(
        model=_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{build_context(results)}\n\nQuestion: {question}"},
        ],
    )
    text = response.choices[0].message.content.strip()
    return None if _REFUSAL_SENTINEL in text else text