from .mapper import ModelMapper
from models import UserProfile

class UserProfileMapper(ModelMapper):
  def to_model(self, document):
    return UserProfile(
      id = str(document['_id']),
      name = document['name'],
      email = document['email'],
      metadata = document['metadata']
    )

    def to_dict(self):
        return {
            'name': self.name,
            'email': self.email,
            'metadata': self.metadata
        }