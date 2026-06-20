from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    author: str = Field(..., description="El ID o email del usuario. Es clave para obtener el perfil y mitigar el problema del arranque en frío (Cold-Start).", examples=["adrianmarino@gmail.com"])
    content: str = Field(..., description="La consulta en lenguaje natural del usuario.", examples=["Quiero ver una película de ciencia ficción de los años 90"])
