import sys
sys.path.append('../lib')

import recommender as rd
from repository import Repository, ConnectionFactory, UserProfileMapper, ChatHistoryMapper
from services import ChatBotService

from langchain.globals import set_debug, set_verbose

set_verbose(False)
set_debug(False)

class AppContext:
  def __init__(self):
    self.connection = ConnectionFactory.create()

    self.profiles_repository = Repository(
        self.connection, 
        'profiles',
        UserProfileMapper(),
        'email'
    )
    self.profiles_repository.add_index('email')

    self.histories_repository = Repository(
        self.connection, 
        'histories',
        ChatHistoryMapper(),
        'email'
    )
    self.histories_repository.add_index('email')

    self.chat_bot = rd.MovieRecommenderChatBotFactory.stateless()

    self.chat_bot_service = ChatBotService(self)