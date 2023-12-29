from fastapi import HTTPException, APIRouter, Response
from models import UserMessage


def chats_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/chats')

  @router.post('')
  async def chat(user_message: UserMessage, metadata: bool = False, simple_mode: bool = False):
      result = await ctx.chat_bot_service.send(user_message)

      if not metadata:
        result.metadata = None

      if simple_mode:
        return Response(content=result.content, media_type="application/txt")
      else:
        return result.dict(exclude_none=True)

  return router