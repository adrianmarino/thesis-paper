from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationQuery
import logging


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations', tags=["Recommendations"])

  @router.get('/models', summary="List available LLM Models")
  async def models():
    """
    Returns the list of Large Language Models (LLMs) that are configured and available in the Ollama environment to be used for generating recommendations.
    """
    return {
      'models': ctx.recommendation_chat_service.available_models()
    }


  @router.post('', summary="Generate Hybrid Recommendations")
  async def recommendations(
    request: Request,
    response: Response,
    query: RecommendationQuery
  ):
    """
    **Hybrid Recommendation Engine (Main Endpoint)**
    
    This endpoint processes a natural language request and generates highly personalized movie recommendations. 
    It combines language understanding from **LLMs**, semantic similarity via **RAG (ChromaDB)**, and user preferences using **Collaborative Filtering (MongoDB)**.

    ### Use Cases
    - **Contextual Recommendation / Exploration**: The user doesn't know what to watch and asks "I want a 90s sci-fi movie". The engine uses the LLM to infer candidates and uses RAG to find the exact movies in the database.
    - **Historical Personalization (Cold-Start mitigated)**: By providing the `author` (email/ID) in the `message` field, the system loads the user's previous ratings and finds similar users (collaborative filtering) to boost or penalize certain movies within the final list.

    ### Internal Execution Flow
    1. **LLM Inference**: A prompt is built by injecting the user's profile (if it exists). The language model returns a first draft of movies.
    2. **RAG Augmentation**: Each title returned by the LLM is searched in ChromaDB to map them to real system IDs and increase variety with semantically very similar items.
    3. **Ranking (Collaborative Filtering)**: The resulting candidates are scored and re-ranked based on the rating predictions of similar users (`k_sim_users`).
    4. **Delivery**: Exactly `recommendations_limit` movies are returned, filtering out those the user has already seen (if `not_seen=True`).

    *Note: If the `plain=True` property is activated in the `settings`, the JSON format is ignored and raw text is returned.*
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
