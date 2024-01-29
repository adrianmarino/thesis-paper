from fastapi import HTTPException, APIRouter, Response, Request
from models import RecommendationsRequest
import sys


def recommenders_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommenders')

  @router.put('/train')
  async def train(
    request: Request,
    response: Response
  ):
    result = ctx.recommender_service.train()
    return result


  return router