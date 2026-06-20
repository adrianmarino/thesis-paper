from pydantic import BaseModel, Field
from .cf_settings import CFSettings
from .rag_settings import RagSettings


class RecommendationSettings(BaseModel):
    llm                     : str  = Field('deepseek-r1:8b', description="Distribución o versión del modelo de lenguaje a utilizar para la generación inicial de candidatos (ej: llama3, deepseek-r1:8b).")
    retry                   : int  = Field(2, description="Cantidad de reintentos en caso de que el LLM falle o devuelva una respuesta malformada.")
    base_url                : str  = Field("", description="URL base de la petición (asignada dinámicamente en tiempo de ejecución).")
    plain                   : bool = Field(False, description="Si es True, la API responderá con texto plano en lugar del JSON estructurado.")
    include_metadata        : bool = Field(False, description="Si es True, incluye en la respuesta metadata adicional sobre el origen de la recomendación (similitudes, etc).")
    rag                     : RagSettings = Field(default_factory=RagSettings)
    collaborative_filtering : CFSettings = Field(default_factory=CFSettings)

    class Config:
        arbitrary_types_allowed = True
