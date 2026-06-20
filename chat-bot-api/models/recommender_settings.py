from pydantic import BaseModel, Field

class RecommenderSettings(BaseModel):
    shuffle                           : bool = Field(False, description="Si es True, mezcla aleatoriamente los resultados obtenidos antes de devolverlos.")
    candidates_limit                  : int  = Field(20, description="Límite máximo de ítems candidatos a evaluar en las bases de datos.")
    llm_response_limit                : int  = Field(20, description="Cantidad máxima de ítems iniciales que el modelo de lenguaje (LLM) intentará generar.")
    recommendations_limit             : int  = Field(5, description="Cantidad final de recomendaciones que se devolverán al usuario.")
    similar_items_augmentation_limit  : int  = Field(5, description="Cantidad de ítems similares a buscar en ChromaDB por cada ítem devuelto por el LLM (RAG).")
