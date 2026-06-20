from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationQuery
import logging


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations', tags=["Recommendations"])

  @router.get('/models', summary="Listar Modelos LLM disponibles")
  async def models():
    """
    Retorna la lista de modelos de lenguaje (LLMs) que están configurados y disponibles en el entorno de Ollama para ser utilizados en la generación de recomendaciones.
    """
    return {
      'models': ctx.recommendation_chat_service.available_models()
    }


  @router.post('', summary="Generar Recomendaciones Híbridas")
  async def recommendations(
    request: Request,
    response: Response,
    query: RecommendationQuery
  ):
    """
    **Motor de Recomendaciones Híbrido (Endpoint Principal)**
    
    Este endpoint procesa una solicitud en lenguaje natural y genera recomendaciones altamente personalizadas de películas. 
    Combina la comprensión del lenguaje de los **LLMs**, similitud semántica mediante **RAG (ChromaDB)** y preferencias del usuario usando **Filtrado Colaborativo (MongoDB)**.

    ### Casos de Uso
    - **Recomendación por contexto/exploración**: El usuario no sabe qué ver y pide "Quiero una película de ciencia ficción de los años 90". El motor utiliza el LLM para inferir candidatos y hace RAG para encontrar las películas exactas en la base de datos.
    - **Personalización Histórica (Cold-Start mitigado)**: Al proveer el `author` (email/ID) en el campo `message`, el sistema carga las calificaciones previas del usuario y busca usuarios afines (filtrado colaborativo) para potenciar o castigar ciertas películas dentro del listado final.

    ### Flujo de Ejecución Interno
    1. **Inferencia LLM**: Se construye un prompt inyectando el perfil del usuario (si existe). El modelo de lenguaje devuelve un primer borrador de películas.
    2. **RAG Augmentation**: Se busca cada título arrojado por el LLM en ChromaDB para mapearlos a IDs reales del sistema e incrementar la variedad con ítems muy similares semánticamente.
    3. **Ranking (Collaborative Filtering)**: Los candidatos resultantes son puntuados y re-ordenados en función de las predicciones de rating de usuarios similares (`k_sim_users`).
    4. **Entrega**: Se devuelven exactamente `recommendations_limit` películas filtrando las que el usuario ya haya visto (si `not_seen=True`).

    *Nota: Si se activa la propiedad `plain=True` en los `settings`, se ignora el formato JSON y se devuelve el texto crudo.*
    """
    query.settings.base_url = str(request.base_url)
    logging.info(f'Query:\n{query.model_dump(exclude_none=True)}')

    recommendations = await ctx.recommendation_chat_service.ask(query)


    if query.settings.plain:
      return Response(content=recommendations.plain, media_type='text/plain')
    else:
      recommendations.items = [item.model_dump(exclude_none=True) for item in recommendations.items]
      return recommendations.model_dump(exclude_none=True)

  return router
