from fastapi import HTTPException, APIRouter, Response
import sys


def chat_histories_handler(base_url, ctx):
    router = APIRouter(prefix=f"{base_url}/histories", tags=["Chat Histories"])

    @router.get("/{email}", summary="Get Chat History by Email")
    async def get_history(email: str):
        """
        Retrieves the conversational context (memory) that the user has maintained with the Chatbot.
        """
        history = await ctx.history_service.find(email)

        if history == None:
            raise HTTPException(status_code=404, detail=f"Not found {email} history")
        else:
            return history

    @router.delete("/{email}", summary="Delete Chat History")
    async def delete_history(email: str):
        """
        Clears the conversational context memory for the given user email.
        """
        await ctx.history_service.delete_by_id(email)
        return Response(status_code=202)

    return router
