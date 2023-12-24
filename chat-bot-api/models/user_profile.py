from .model import Model
import typing

class UserProfile(Model):
    name : str
    email : str
    metadata: typing.Dict[str, typing.Any]