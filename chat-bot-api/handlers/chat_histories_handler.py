from fastapi import HTTPException, APIRouter, Response


def chat_histories_handler(base_url, ctx):
    router = APIRouter(prefix=f"{base_url}/histories")

    @router.get("/{email}")
    async def get_history(email: str):
        history = await ctx.history_service.find(email)

        if history == None:
            raise HTTPException(status_code=404, detail=f"Not found {email} history")
        else:
            return history

    @router.delete("/{email}")
    async def delete_history(email: str):
        await ctx.history_service.delete_by_id(email)
        return Response(status_code=202)

    return router
