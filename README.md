# Backend RAG

Backend FastAPI para orquestrar RAG (ingestão, busca semântica, LLM via Vitoria-AI)

## Arquitetura

- FastAPI expõe API institucional (ingestão, consulta, saúde, geração IA).
- Orquestrador RAG monta contexto: chunking simples, embeddings determinísticos, busca vetorial em memória, chamada à LLM.
- LLM via Vitoria-AI (HTTP chat/completions). A LLM nunca acessa dados direto; só recebe o contexto montado pelo RAG.
- Bases: vetorial em memória (substituível por FAISS/PGVector), metadados em memória (substituível por SQL), storage externo para originais (futuro).

Fluxo resumido:

1. Ingestão: documento → chunking → embeddings → base vetorial.
2. Consulta: pergunta → validação leve → busca semântica → contexto → LLM → resposta + fontes.

## Endpoints principais

- `GET /health`: liveness.
- `POST /ingest`: ingere documento bruto com metadados opcionais.
- `POST /query`: roda pipeline RAG e devolve resposta e fontes.
- `POST /ai/generate`: expõe geração direta via LLM.

## Rodando localmente

1. Crie e ative um venv:
   - macOS/Linux: `python -m venv .venv && source .venv/bin/activate`
   - Windows (PowerShell): `python -m venv .venv; .venv\\Scripts\\Activate.ps1`
2. Instale dependências (com o venv ativo): `pip install -r requirements.txt`.
3. Suba o servidor Vitoria-AI (local em `../vitoria-ai`): `uvicorn main:app --reload --port 9000`.
4. Configure `.env` (veja abaixo) com `VITORIA_BASE_URL`, `VITORIA_MODEL` e, se houver, `VITORIA_API_KEY`.
5. Inicie a API usando o binário do venv:
   - `.venv/bin/uvicorn app.main:app --reload`
   - ou, se o venv estiver ativado: `uvicorn app.main:app --reload`
6. Swagger: `http://localhost:8000/docs` (API institucional) e `http://localhost:8000/redoc`.

Reabrindo o terminal (sem reinstalar deps):

```bash
cd /Users/bryanneves/Documents/indt/VITORIA/rag-backend
source .venv/bin/activate
.venv/bin/uvicorn app.main:app --reload
```

Variáveis suportadas (ver `.env.example`):

- `APP_NAME`, `APP_VERSION`, `VECTOR_DIM`, `MAX_CONTEXT_CHARS`, `TEMPERATURE`.
- `VITORIA_BASE_URL` (ex.: `http://localhost:9000`).
- `VITORIA_API_KEY` (se o endpoint exigir bearer token).
- `VITORIA_MODEL` (nome do modelo no servidor Vitoria-AI).

## Próximos passos sugeridos

- Ajustar políticas de uso do modelo (Azure OpenAI, etc.) e quotas.
- Persistência real: banco relacional para metadados/logs e FAISS/PGVector para vetores.
- Autenticação/autorização, trilhas de auditoria, quotas, PII guardrails.
- Avaliação automática das respostas (factualidade, cobertura de fontes).
