import sys
sys.path.append('../lib')

from mappers import (
    UserProfileMapper,
    ChatHistoryMapper,
    InteractionMapper,
    ItemMapper,
    ItemContentEmbMapper,
    EntityEmbMapper
)
from repository.mongo import (
    MongoRepository,
    MongoConnectionFactory
)
from repository.chroma import (
    ChromaRepositoryFactory
)
from services import (
    RecommendationChatService,
    ChatHistoryService,
    ProfileService,
    InteractionService,
    ItemService,
    SentenceEmbeddingService,
    OllamaSentenceEmbeddingService,
    InteractionInfoService,
    ChatBotPoolService,
    RecommendationsFactory,
    RecommenderService
)

from jobs import CFEmbUpdateJob
from recommenders import DatabaseUserItemFilteringRecommender


class AppContext:
    def __init__(self):
        self._build_mongo_repositories()
        self._build_chroma_repositories()
        self._build_services()
        self._build_jobs()
        self._build_recommenders()


    def _build_recommenders(self):
        self.database_user_item_filtering_recommender = DatabaseUserItemFilteringRecommender(
            user_emb_repository          = self.users_cf_emb_repository,
            items_repository             = self.items_repository,
            interactions_repository      = self.interactions_repository,
            pred_interactions_repository = self.pred_interactions_repository,
            item_service                 = self.item_service
        )


    def _build_services(self):
        self.chat_bot_pool_service = ChatBotPoolService()

        self.recommendation_chat_service = RecommendationChatService(self)

        self.history_service = ChatHistoryService(self)

        self.profile_service = ProfileService(self)

        self.interaction_service = InteractionService(self)

        self.item_service = ItemService(self)

        self.recommender_service = RecommenderService(self)


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
        self.interactions_repository.add_multi_index(['item_id'], unique=False)


        self.pred_interaction_mapper = InteractionMapper()
        self.pred_interactions_repository = MongoRepository(
            self.mongo_connection,
            'pred_interactions',
            self.pred_interaction_mapper,
            'user_id'
        )
        self.pred_interactions_repository.add_multi_index(['user_id', 'item_id'])
        self.pred_interactions_repository.add_multi_index(['item_id'], unique=False)


        self.item_mapper = ItemMapper()
        self.items_repository = MongoRepository(
            self.mongo_connection,
            'items',
            self.item_mapper,
            'item_id'
        )
        self.items_repository.add_single_index('item_id')
        self.items_repository.add_single_index('user_id', unique=False)

        self.interaction_info_service = InteractionInfoService(self)

        self.recommendations_factory = RecommendationsFactory(self)


    def _build_chroma_repositories(self):
        self.sentence_emb_service = SentenceEmbeddingService(model='all-mpnet-base-v2')
        # self.sentence_emb_service = OllamaSentenceEmbeddingService(model='mxbai-embed-large')
        # self.sentence_emb_service = OllamaSentenceEmbeddingService(model='llama3:text')

        self.item_emb_mapper = ItemContentEmbMapper(self.sentence_emb_service)
        self.entity_emb_mapper = EntityEmbMapper()

        self.repo_factory = ChromaRepositoryFactory()

        self.items_content_emb_repository = self.repo_factory.create(
            'items_content',
            self.item_emb_mapper
        )

        self.items_cf_emb_repository = self.repo_factory.create(
            'items_cf',
            self.entity_emb_mapper
        )

        self.users_cf_emb_repository = self.repo_factory.create(
            'users_cf',
            self.entity_emb_mapper
        )


    def _build_jobs(self):
        self.cf_emb_update_job = CFEmbUpdateJob(self)