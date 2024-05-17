from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationQuery


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.get('/models')
  async def models():
    return {
      'models': ctx.recommendation_chat_service.available_models()
    }


  @router.post('')
  async def recommendations(
    request: Request,
    response: Response,
    query: RecommendationQuery
  ):
    query.settings.base_url = str(request.base_url)

    recommendations = await ctx.recommendation_chat_service.ask(query)

    if query.settings.plain:
      return Response(content=recommendations.plain, media_type='text/plain')
    else:
      recommendations.items = [item.model_dump(exclude_none=True) for item in recommendations.items]
      return recommendations.model_dump(exclude_none=True)

  return router