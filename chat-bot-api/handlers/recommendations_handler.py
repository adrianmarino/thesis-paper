from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationsRequest
import sys


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.get('/models')
  async def models():
    return {
      'models': ctx.chat_bot_service.available_models()
    }


  @router.post('')
  async def recommendations(
    request: Request,
    response: Response,
    recommendations_request: RecommendationsRequest
  ):
    settings = recommendations_request.recommender_settings

    while True:
      recommendations = await ctx.chat_bot_service.send(
        recommendations_request.message,
        model            = settings.model,
        base_url         = str(request.base_url),
        include_metadata = settings.metadata,
        shuffle          = settings.shuffle,
        candidates_limit = settings.candidates_limit,
        parse_limit      = settings.parse_limit,
        result_limit     = settings.limit
      )
      if not recommendations.empty or settings.retry == 0:
        break
      retry -= 1

    if settings.plain:
      return Response(content=recommendations.plain, media_type='text/plain')
    if settings.metadata:
      return recommendations.dict(exclude_none=True)
    else:
      return [item.dict(exclude_none=True) for item in recommendations.items]


  return router