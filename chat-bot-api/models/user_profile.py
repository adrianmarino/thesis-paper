from .model import Model
import typing
from bunch import Bunch


class UserProfile(Model):
    name : str
    email : str
    metadata: typing.Dict[str, typing.Any]


    def __str__(self):
        text = f'\n- Name: {self.name}'

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

        prefered_movies = Bunch(metadata.prefered_movies)

        if prefered_movies:
            text += f'\n- Movie preferences:'

            if 'from' in prefered_movies.release:
                text += f'\n  - Movies released from the {prefered_movies.release["from"]} year'

            if prefered_movies.genres:
                text += f'\n  - Movies that have the following genres: {", ".join(prefered_movies.genres)}'

        return text