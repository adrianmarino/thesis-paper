from .recommender_settings import RecommenderSettings
from pydantic import Field

class CFSettings(RecommenderSettings):
    text_query_limit               : int   = Field(2000, description="Límite de caracteres para la búsqueda textual en filtrado colaborativo.")
    not_seen                       : bool  = Field(True, description="Si es True, excluye de las recomendaciones aquellas películas que el usuario ya ha calificado.")
    k_sim_users                    : int   = Field(5, description="Cantidad de usuarios similares (vecinos más cercanos) a tener en cuenta para calcular predicciones.")
    random_selection_items_by_user : float = Field(0.5, description="Porcentaje de ítems seleccionados aleatoriamente del historial de usuarios similares (aporta diversidad).")
    max_items_by_user              : int   = Field(5, description="Límite de ítems a extraer del historial de cada usuario similar.")
    min_rating_by_user             : float = Field(3.5, description="Calificación mínima que debe tener un ítem en el historial del usuario similar para ser considerado un buen candidato.")
    rank_criterion                 : str   = Field('user_sim_weighted_pred_rating_score', description="Criterio de reordenamiento de los candidatos. Opciones: user_sim_weighted_rating_score, user_item_sim, pred_user_rating")
