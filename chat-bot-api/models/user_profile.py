from pydantic import BaseModel
import typing
from bunch import Bunch


class UserProfile(BaseModel):
    name : str
    email : str
    metadata: typing.Dict[str, typing.Any]


    @property
    def release_from(self):
        return int(self.metadata['preferred_movies']['release']['from'].strip())


    @property
    def genres(self):
        return self.metadata['preferred_movies']['genres']


    def __str__(self):
        text = f'User profile:\n'
        text += f'- Name: {self.name}'

        metadata = Bunch(self.metadata)

        if metadata.genre:
            text += f'\n- Genre: {"Male" if metadata.genre.lower() == "male" else "Female"}'

        if metadata.age:
            text += f'\n- Age: {metadata.age}'

        if metadata.nationality:
            text += f'\n- Nationality: {metadata.nationality}'

        if metadata.studies:
            text += f'\n- Studies: {metadata.studies}'

        if metadata.work:
            text += f'\n- Work: {metadata.work}'

        preferred_movies = Bunch(metadata.preferred_movies)

        if preferred_movies:
            text += f'\n- Movie preferences:'

            if 'from' in preferred_movies.release:
                text += f'\n  - Released from {preferred_movies.release["from"]}'

            if preferred_movies.genres:
                text += f'\n  - Genres: {", ".join(preferred_movies.genres)}'

        return text