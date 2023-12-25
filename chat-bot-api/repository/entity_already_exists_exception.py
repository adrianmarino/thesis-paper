class EntityAlreadyExistsException(Exception):
    def __init__(self, cause):
        self.message = f'Entity already exists'
        self.cause   = cause
        super().__init__(self.message)