from fastapi import APIRouter, Depends

from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.llm import VitoriaLLMClient
from app.container import get_llm_client

router = APIRouter(prefix="/ai", tags=["ia"])


@router.post("/generate", response_model=GenerateResponse)
def generate(
    payload: GenerateRequest,
    llm: VitoriaLLMClient = Depends(get_llm_client),
) -> GenerateResponse:
    output = llm.generate(payload.prompt, context_chunks=[], max_tokens=payload.max_tokens)
    return GenerateResponse(output=output)
