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
                    "seja curta (2-4 frases) e cite apenas trechos relevantes do contexto. "
                    "Se o contexto não trouxer nada útil ou a pergunta for fora do tema, diga que não há "
                    "informação relevante e não invente. Não repita o contexto inteiro, apenas referencie. "
                    "Retorne somente a resposta final, sem repetir instruções ou o contexto."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Pergunta: {question}\n\nContexto:\n{context}\n\n"
                    "Responda de forma objetiva, só usando o que houver de relevante no contexto. "
                    "Não repita o contexto; apenas responda."
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
            content = message.get("content") or ""
            cleaned = self._clean_response(content)
            return cleaned or "Resposta vazia retornada pela Vitoria-AI."
        except Exception as exc:  # pragma: no cover - rede externa
            return f"Falha ao chamar Vitoria-AI: {exc}"

    @staticmethod
    def _clean_response(content: str) -> str:
        text = content.strip()
        # Se o modelo ecoar prompt (contendo Pergunta/Contexto), tente pegar só o último bloco
        if "Pergunta:" in text or "Contexto:" in text:
            parts = [p.strip() for p in text.split("\n\n") if p.strip()]
            if parts:
                text = parts[-1]
        return text
