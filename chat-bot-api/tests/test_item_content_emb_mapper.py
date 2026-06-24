import pytest
import sys
import os
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mappers.item_content_emb_mapper import ItemContentEmbMapper
from models.item import Item


def test_item_content_emb_mapper_to_params():
    item_service_mock = MagicMock()
    item_service_mock.generate.return_value = np.array([[0.1, 0.2, 0.3]])

    mapper = ItemContentEmbMapper(item_service=item_service_mock)

    item = Item(
        id="movie1",
        imdb_id="tt123",
        title="Inception",
        description="A dream within a dream.",
        release="2010",
        genres=["Action", "Sci-Fi"],
        rating=4.8,
        poster="https://example.com/poster.jpg"
    )

    params = mapper.to_params([item])

    assert params.embeddings == [[0.1, 0.2, 0.3]]
    assert params.ids == ["movie1"]
    assert len(params.metadatas) == 1
    assert params.metadatas[0] == {
        'id': 'movie1',
        'release': 2010,
        'genres': 'Action,Sci-Fi',
        'rating': 4.8
    }
    assert params.documents == ["Title: Inception. Description: A dream within a dream.. Genres: Action, Sci-Fi."]
    item_service_mock.generate.assert_called_once_with(params.documents)


def test_item_content_emb_mapper_to_params_none_rating():
    item_service_mock = MagicMock()
    item_service_mock.generate.return_value = np.array([[0.1, 0.2, 0.3]])

    mapper = ItemContentEmbMapper(item_service=item_service_mock)

    # Omit rating parameter to test default None mapping
    item = Item(
        id="movie2",
        imdb_id="tt456",
        title="Interstellar",
        description="Space exploration.",
        release="2014",
        genres=["Adventure", "Drama"],
        poster="https://example.com/poster.jpg"
    )

    params = mapper.to_params([item])

    assert params.metadatas[0]['rating'] == 0.0
