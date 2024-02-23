import pytorch_common.util as pu
import util as ut
import numpy as np
import logging
import pandas as pd
from .database_user_item_filtering_recommender_result  import DatabaseUserItemFilteringRecommenderResult
import random



class DatabaseUserItemFilteringRecommender:
    def __init__(
        self,
        user_emb_repository,
        items_repository,
        interactions_repository,
        pred_interactions_repository
    ):
        self.__user_emb_repository          = user_emb_repository
        self.__items_repository             = items_repository
        self.__interactions_repository      = interactions_repository
        self.__pred_interactions_repository = pred_interactions_repository


    def __users_distance(self, similar_users):
        return { similar_users.str_ids[idx]: similar_users.distances[idx] for idx in range(len(similar_users.str_ids)) }            

    
    
    def __select_interactions(self, interactions, percent, max_items_by_user):
        interactions_by_user_id = {}
        for i in interactions:
            if i.user_id not in interactions_by_user_id:                
                interactions_by_user_id[i.user_id] = []

            inters = interactions_by_user_id[i.user_id]
                
            if random.random() >= percent and len(inters) < max_items_by_user:
                inters.append(i)
    
        interactions = []
        for inters in interactions_by_user_id.values():
            interactions.extend(inters)
        
        return interactions
    
    
    async def recommend(
        self,
        user_id                        : int  = None,
        not_seen                       : bool = True,
        k_sim_users                    : int = 5,
        random_selection_items_by_user : int = 0.5,
        max_items_by_user              : int = 5
    ):
        similar_users = self.__user_emb_repository.find_similars_by_id(user_id, limit=k_sim_users)
        
        if similar_users.empty: return []

        similar_user_ids = [id for id in similar_users.str_ids if id != user_id]

        if len(similar_user_ids) ==0:
            logging.warning('Not found similar users')
            return []

        interactions = await self.__interactions_repository.find_many_by(user_id={'$in': similar_user_ids})
        
        interactions = self.__select_interactions(
            interactions,
            percent           = random_selection_items_by_user,
            max_items_by_user = max_items_by_user
        )
    
        if len(interactions) ==0:
            logging.warning('Not found sumilar users interactions')
            return []
        
        item_ids = np.unique([i.item_id for i in interactions]).tolist()
        
        if len(item_ids) ==0:
            logging.warning('Not found items')
            return []

        user_interactions = await self.__interactions_repository.find_many_by(user_id=user_id)
 
        if not_seen:
            seen_item_ids = [i.item_id for i in user_interactions]
            item_ids = [item_id for item_id in item_ids if item_id not in seen_item_ids]

        
        items = await self.__items_repository.find_many_by(item_id={'$in': item_ids})
    
        pred_interactions =  await self.__pred_interactions_repository.find_many_by(
            user_id=user_id,
            item_id={'$in': item_ids}
        )
        pred_rating_by_item_id = {i.item_id: i.rating for i in pred_interactions}
        
        
        distance_by_user_id = self.__users_distance(similar_users)
        
        distance_by_item_id = {i.item_id:distance_by_user_id[i.user_id] for i in interactions}
        
        max_rating      = np.max([item.rating for item in items])
        
        max_pred_rating = np.max(list(pred_rating_by_item_id.values()))


        scored_items  = []
        for item in items:
            item_sim   = (1 - distance_by_item_id[item.id])

            norm_rating = item.rating / max_rating

            item_score1  = norm_rating * item_sim
            
            norm_pred_rating = pred_rating_by_item_id.get(item.id, 0) / max_pred_rating
                
            item_score2  = norm_pred_rating * item_sim
            
            scored_items.append((item, item_score1, item_score2, item_sim, norm_rating))

    
        recommended_items = pd.DataFrame([
            {
                'id'    : item[0].id,
                'user_sim_weighted_rating_score'      : item[1],
                'user_sim_weighted_pred_rating_score' : item[2],
                'user_item_sim'                       : item[3],
                'pred_user_rating'                    : pred_rating_by_item_id.get(item[0].id, 0),
                'rating': item[0].rating,
                'title' : item[0].title,
                'poster': item[0].poster,
                'genres': item[0].genres
            }
            for item in scored_items
        ])
        
        
        seen_item_rating_by_id = {i.item_id: i.rating for i in user_interactions}
        seen_items = await self.__items_repository.find_many_by(
            item_id={'$in': list(seen_item_rating_by_id.keys())}
        )
   
        seen_items = pd.DataFrame([
            {
                'id'    : item.id,
                'rating': seen_item_rating_by_id.get(item.id, 0),
                'title' : item.title,
                'poster': item.poster,
                'genres': item.genres
            }
            for item in seen_items
        ])
        
        return DatabaseUserItemFilteringRecommenderResult(
            self.__class__.__name__,
            recommended_items,
            seen_items
        )