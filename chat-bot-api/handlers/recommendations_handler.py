from fastapi import HTTPException, APIRouter, Response, Request
from models import UserMessage


def recommendations_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommendations')

  @router.post('')
  async def recommendations(
    request: Request,
    response: Response,
    user_message: UserMessage,
    plain: bool = False,
    metadata: bool = False
  ):
    recommendations = await ctx.chat_bot_service.send(
      user_message,
      base_url=str(request.base_url),
      include_metadata=metadata
    )

    if plain:
      return Response(content=recommendations.plain, media_type='text/plain')
    else:
      return recommendations.items


  return router