import pytest
import sys
import os
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from recommenders.database_user_item_filtering_recommender import DatabaseUserItemFilteringRecommender
from models.user_interaction import UserInteraction
from models.item import Item


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_database_user_item_filtering_recommender_find_items_not_seen():
    user_emb_repository = MagicMock()
    items_repository = MagicMock()
    interactions_repository = MagicMock()
    pred_interactions_repository = MagicMock()
    item_service = MagicMock()

    recommender = DatabaseUserItemFilteringRecommender(
        user_emb_repository=user_emb_repository,
        items_repository=items_repository,
        interactions_repository=interactions_repository,
        pred_interactions_repository=pred_interactions_repository,
        item_service=item_service
    )

    sim_user_interactions = [
        UserInteraction(user_id="user2", item_id="movie1", rating=5.0),
        UserInteraction(user_id="user3", item_id="movie2", rating=4.5)
    ]

    user_interactions = [
        UserInteraction(user_id="user1", item_id="movie1", rating=4.0)
    ]

    interactions_repository.find_many_by = AsyncMock(return_value=user_interactions)

    items_repository.find_many_by = AsyncMock(return_value=[
        Item(id="movie2", imdb_id="tt2", title="M2", description="Desc", release="2020", genres=["Sci-Fi"], rating=4.5, poster="p")
    ])

    items, item_ids = await recommender._DatabaseUserItemFilteringRecommender__find_items(
        user_id="user1",
        sim_user_interactions=sim_user_interactions,
        not_seen=True,
        text_query=None
    )

    assert item_ids == ["movie2"]
    assert len(items) == 1
    assert items[0].id == "movie2"

    interactions_repository.find_many_by.assert_called_with(user_id="user1")
    items_repository.find_many_by.assert_called_with(item_id={'$in': ["movie2"]})
