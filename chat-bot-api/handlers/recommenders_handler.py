from fastapi import HTTPException, APIRouter, Response, Request


def recommenders_handler(base_url, ctx):
  router = APIRouter(prefix=f'{base_url}/recommenders', tags=["Recommenders (CF)"])

  @router.put('/train', summary="Trigger Recommenders Training")
  async def train(
    request: Request,
    response: Response
  ):
    """
    Manually triggers the training and synchronization process of the Collaborative Filtering recommenders.
    Use this if you do not want to wait for the next scheduled Airflow DAG execution.
    """
    result = ctx.recommender_service.train()
    return result


  return router
