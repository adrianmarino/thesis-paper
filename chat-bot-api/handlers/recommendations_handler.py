from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationQuery, Recommendations
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


  @router.post('', response_model=Recommendations, summary="Generate Hybrid Recommendations")
  async def recommendations(
    request: Request,
    response: Response,
    query: RecommendationQuery
  ):
    """
    **Hybrid Recommendation Engine (Main Endpoint)**
    
    This endpoint processes a natural language request and generates highly personalized movie recommendations. 
    It operates as a Retrieve-and-Generate pipeline, where databases act as the retrieval engine and Large Language Models (LLMs) act as the final personalized filter.

    ### Execution Flow
    1. **Candidate Retrieval (RAG & Collaborative Filtering):** 
       First, the system fetches a bounded list of candidate movies based on the user's interaction history count.
       - *Cold-Start (< 20 ratings)*: The engine uses **RAG (Semantic Search)** against ChromaDB to find movies matching the text query and the release-date preferences of the user's profile.
       - *Warm-Start (>= 20 ratings)*: The engine uses **Collaborative Filtering**, finding items rated highly by similar users (Nearest Neighbors) and scoring them against the user's query.
    2. **LLM Context Injection & Selection:** 
       The candidate movies obtained in Step 1 are passed as context to the chosen **LLM** (e.g., Llama3). The LLM also receives the user's full profile, their previously watched movies, and their natural language question. The LLM acts as the final judge, filtering and selecting the most appropriate movies from the candidates.
    3. **Resolution & Augmentation:** 
       The movies selected by the LLM are mapped back to the database. If configured (`similar_items_augmentation_limit`), the system will append semantically similar items to the final response to guarantee diversity before returning the JSON.

    ### Body Parameters Highlights
    - **`message.author`**: The user's email. Mandatory to load historical data and profiles.
    - **`message.content`**: The question itself (e.g., "I want a 90s sci-fi movie").
    - **`settings.llm`**: Choose the underlying language model to use as the final filter.
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
