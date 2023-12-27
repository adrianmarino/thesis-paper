import sys
sys.path.append('../lib')

import recommender as rd
from mappers import (
    UserProfileMapper,
    ChatHistoryMapper,
    InteractionMapper,
    ItemMapper
)
from repository import Repository, ConnectionFactory
from services import (
    ChatBotService,
    ChatHistoryService,
    ProfileService,
    InteractionService,
    ItemService
)


class AppContext:
  def __init__(self):
    self.connection = ConnectionFactory.create()


    self.profile_mapper = UserProfileMapper()
    self.profiles_repository = Repository(
        self.connection,
        'profiles',
        self.profile_mapper,
        'email'
    )
    self.profiles_repository.add_single_index('email')


    self.history_mapper = ChatHistoryMapper()
    self.histories_repository = Repository(
        self.connection,
        'histories',
        self.history_mapper,
        'email'
    )
    self.histories_repository.add_single_index('email')


    self.interaction_mapper = InteractionMapper()
    self.interactions_repository = Repository(
        self.connection,
        'interactions',
        self.interaction_mapper,
        'user_id'
    )
    self.interactions_repository.add_multi_index(['user_id', 'item_id'])


    self.item_mapper = ItemMapper()
    self.items_repository = Repository(
        self.connection,
        'items',
        self.item_mapper,
        'item_id'
    )
    self.items_repository.add_single_index('item_id')


    self.chat_bot = rd.MovieRecommenderChatBotFactory.stateless()

    self.chat_bot_service = ChatBotService(self)

    self.history_service = ChatHistoryService(self)

    self.profile_service = ProfileService(self)

    self.interaction_service = InteractionService(self)

    self.item_service = ItemService(self)
