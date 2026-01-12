from typing import List, Optional

from openai import OpenAI


class LLMClient:
    """Cliente de LLM baseado em OpenAI; mantém contrato simples de geração."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url) if api_key else None

    def generate(self, question: str, context_chunks: List[str], max_tokens: int = 256) -> str:
        if not context_chunks:
            return "Nenhum contexto encontrado para responder."

        if not self.client:
            return "LLM OpenAI não configurada (defina OPENAI_API_KEY)."

        context = "\n---\n".join(context_chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é um assistente do Ministério Público. Responda em português, "
                    "cite fatos do contexto e mantenha concisão. Não invente fontes."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pergunta: {question}\n\nContexto:\n{context}\n\n"
                    "Responda de forma objetiva, citando o que vier do contexto."
                ),
            },
        ]

        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - rede externa
            return f"Falha ao chamar LLM: {exc}"
