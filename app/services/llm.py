from typing import List, Optional

import httpx


class VitoriaLLMClient:
    """Cliente HTTP para a Vitoria-AI (contrato compatível com chat/completions)."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str],
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = timeout

    def generate(self, question: str, context_chunks: List[str], max_tokens: int = 256) -> str:
        if not self.base_url:
            return "Vitoria-AI não configurada (defina VITORIA_BASE_URL)."

        context = "\n---\n".join(context_chunks) if context_chunks else "(Sem contexto fornecido)"
        messages = [
            {
                "role": "system",
                "content": (
                    "Você é a Vitoria-AI, assistente proprietária. Responda em português, "
                    "cite trechos do contexto e seja conciso. Não invente fontes."
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

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            url = f"{self.base_url}/v1/chat/completions"
            response = httpx.post(url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices") or []
            first = choices[0] if choices else {}
            message = first.get("message") or {}
            content = message.get("content")
            return content or "Resposta vazia retornada pela Vitoria-AI."
        except Exception as exc:  # pragma: no cover - rede externa
            return f"Falha ao chamar Vitoria-AI: {exc}"
