import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.item_sim_query import ItemSimQuery


def test_item_sim_query_rating_gte_float():
    query = ItemSimQuery().rating_gte(3.5)
    assert query.rating == 3.5
    assert query.where == {'rating': {'$gte': 3.5}}


def test_item_sim_query_rating_gte_int():
    query = ItemSimQuery().rating_gte(4)
    assert query.rating == 4.0
    assert query.where == {'rating': {'$gte': 4.0}}


def test_item_sim_query_rating_gte_invalid():
    query = ItemSimQuery().rating_gte(-1.0)
    assert query.rating == 0.0
    assert query.where == {}


def test_item_sim_query_rating_gte_none():
    query = ItemSimQuery().rating_gte(None)
    assert query.rating == 0.0
    assert query.where == {}


def test_item_sim_query_multiple_conditions():
    query = ItemSimQuery().release_gte(2020).rating_gte(4.5)
    # The order of dictionary items in conditions might vary, but in python 3.7+ dictionaries preserve insertion order.
    # We inserted 'release' first, then 'rating'.
    assert query.where == {
        '$and': [
            {'release': {'$gte': 2020}},
            {'rating': {'$gte': 4.5}}
        ]
    }
