import sys
sys.path.append('../lib')

import recommender as rd
from mappers import (
    UserProfileMapper,
    ChatHistoryMapper,
    InteractionMapper,
    ItemMapper,
    ItemEmbMapper
)
from repository.mongo import (
    MongoRepository,
    MongoConnectionFactory
)
from repository.chroma import (
    ChromaRepositoryFactory
)
from services import (
    ChatBotService,
    ChatHistoryService,
    ProfileService,
    InteractionService,
    ItemService,
    EmbService
)
from chromadb import PersistentClient


class AppContext:
    def __init__(self):
        self._build_mongo_repositories()
        self._build_chroma_repositories()
        self._build_services()


    def _build_services(self):
        self.chat_bot = rd.MovieRecommenderChatBotFactory.stateless()

        self.chat_bot_service = ChatBotService(self)

        self.history_service = ChatHistoryService(self)

        self.profile_service = ProfileService(self)

        self.interaction_service = InteractionService(self)

        self.item_service = ItemService(self)

    def _build_mongo_repositories(self):
        self.mongo_connection = MongoConnectionFactory.create()

        self.profile_mapper = UserProfileMapper()
        self.profiles_repository = MongoRepository(
            self.mongo_connection,
            'profiles',
            self.profile_mapper,
            'email'
        )
        self.profiles_repository.add_single_index('email')


        self.history_mapper = ChatHistoryMapper()
        self.histories_repository = MongoRepository(
            self.mongo_connection,
            'histories',
            self.history_mapper,
            'email'
        )
        self.histories_repository.add_single_index('email')


        self.interaction_mapper = InteractionMapper()
        self.interactions_repository = MongoRepository(
            self.mongo_connection,
            'interactions',
            self.interaction_mapper,
            'user_id'
        )
        self.interactions_repository.add_multi_index(['user_id', 'item_id'])


        self.item_mapper = ItemMapper()
        self.items_repository = MongoRepository(
            self.mongo_connection,
            'items',
            self.item_mapper,
            'item_id'
        )
        self.items_repository.add_single_index('item_id')


    def _build_chroma_repositories(self):
        self.emb_service = EmbService('all-roberta-large-v1')

        self.item_emb_mapper = ItemEmbMapper(self.emb_service)

        client = PersistentClient(path='.')

        self.items_emb_repository = ChromaRepositoryFactory().create(
            'items',
            self.item_emb_mapper
        )