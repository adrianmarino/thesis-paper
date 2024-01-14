from models import UserMessage, AIMessage, ChatHistory, UserInteractionInfo
import util as ut

class InteractionInfoService:
  def __init__(self, ctx):
    self.ctx = ctx


  async def find_by_user_id(self, user_id: str):
    interactions = await self.ctx.interaction_service.find_by_user_id(user_id)
    items        = await self.ctx.item_service.find_by_ids([i.item_id for i in interactions])
    return [UserInteractionInfo(interaction=int, item=item) for int, item in zip(interactions, items)]
