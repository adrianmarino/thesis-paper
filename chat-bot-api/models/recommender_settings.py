from pydantic import BaseModel, Field

class RecommenderSettings(BaseModel):
    shuffle                           : bool = Field(False, description="If True, randomly shuffles the retrieved results before returning them.")
    candidates_limit                  : int  = Field(20, description="Maximum number of candidate items to evaluate from the databases.")
    llm_response_limit                : int  = Field(20, description="Maximum number of initial items that the Large Language Model (LLM) will attempt to generate.")
    recommendations_limit             : int  = Field(5, description="Final number of recommendations that will be returned to the user.")
    similar_items_augmentation_limit  : int  = Field(5, description="Number of similar items to search for in ChromaDB for each item returned by the LLM (RAG).")
