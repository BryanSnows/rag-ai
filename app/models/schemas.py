from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    content: str = Field(..., description="Documento bruto a ser ingestado")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadados opcionais")
    source_id: Optional[str] = Field(default=None, description="Identificador opcional do documento original")

    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "Relatório de diligência: foi realizada inspeção no local em 10/01/2026. Foram coletados depoimentos de três testemunhas.",
                "metadata": {
                    "tipo": "relatorio",
                    "processo": "MP-2026-12345",
                    "data": "2026-01-10",
                },
                "source_id": "relatorio-2026-01-10",
            }
        }
    }


class DocumentIngestResponse(BaseModel):
    document_id: str
    segments: int


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quando foi a inspeção e quantas testemunhas foram ouvidas?",
                "top_k": 3,
            }
        }
    }


class SourceAttribution(BaseModel):
    document_id: str
    score: float
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceAttribution]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=256, gt=0, le=1024)

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "Liste os próximos passos para um relatório de diligência do MP.",
                "max_tokens": 150,
            }
        }
    }


class GenerateResponse(BaseModel):
    output: str
