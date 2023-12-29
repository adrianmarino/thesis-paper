from .model import Model
import typing
from bunch import Bunch


class UserProfile(Model):
    name : str
    email : str
    metadata: typing.Dict[str, typing.Any]


    def __str__(self):
        text = f'Perfil del usuario:\n\n- Nombre: {self.name}'

        metadata = Bunch(self.metadata)

        if metadata.genre:
            text += f'\n- Género: {"Masculino" if metadata.genre.lower() == "male" else "Femenino"}'

        if metadata.age:
            text += f'\n- Edad: {metadata.age}'

        if metadata.nationality:
            text += f'\n- Nacionalidad: {metadata.nationality}'

        if metadata.studies:
            text += f'\n- Estudios: {metadata.studies}'

        if metadata.work:
            text += f'\n- Ocupación: {metadata.work}'

        prefered_movies = Bunch(metadata.prefered_movies)

        if prefered_movies:
            text += f'\n- Preferencias de películas:'

            if 'from' in prefered_movies.release:
                text += f'\n  - Películas estrenadas a partir del año {prefered_movies.release["from"]}'

            if prefered_movies.genres:
                text += f'\n  - Películas que tengan los siguientes géneros: {", ".join(prefered_movies.genres)}'

        return text