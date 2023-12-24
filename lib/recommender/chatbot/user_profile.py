class UserProfile:
  def __init__(
    self,
    name,
    email,
    metadata,
    id=None,
    history=[],
    interactions=[]
  ):
    self.id           = id
    self.name         = name
    self.email        = email
    self.metadata     = metadata
    self.history      = history
    self.interactions = interactions