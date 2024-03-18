import attr
import json
import attr.validators




class Dto:
    def to_json(self):
      return json.dumps(
        self._dict(),
        indent    = 2,
        sort_keys = False
      )

@attr.s
class UserProfileDto(Dto):
    name             = attr.ib(type=str)
    email            = attr.ib(type=str)
    studies          = attr.ib(type=str, default=None)
    age              = attr.ib(type=int, default=None)
    genre            = attr.ib(type=str, default=None)
    nationality      = attr.ib(type=str, default=None)
    work             = attr.ib(type=str, default=None)
    preferred_from   = attr.ib(type=str, default=None)
    preferred_genres = attr.ib(type=list[str], default=[])



    def _dict(self):
      return {
        'name'    : self.name,
        'email'   : self.email,
        'metadata': {
          'studies'          : self.studies,
          'age'              : self.age,
          'genre'            : self.genre,
          'nationality'      : self.nationality,
          'work'             : self.work,
          'preferred_movies' :  {
              'release': {
                'from': self.preferred_from,
              },
              'genres' : self.preferred_genres
          }
        }
      }

    @staticmethod
    def from_json(data):
      return UserProfileDto(
        name              = data['name'],
        email             = data['email'],
        studies           = data['metadata']['studies']                             if 'metadata' in data else None,
        age               = data['metadata']['age']                                 if 'metadata' in data else None,
        genre             = data['metadata']['genre']                               if 'metadata' in data else None,
        nationality       = data['metadata']['nationality']                         if 'metadata' in data else None,
        work              = data['metadata']['work']                                if 'metadata' in data else None,
        preferred_from    = data['metadata']['preferred_movies']['release']['from'] if 'metadata' in data and 'preferred_movies' in data['metadata'] else None,
        preferred_genres  = data['metadata']['preferred_movies']['genres']          if 'metadata' in data and 'preferred_movies' in data['metadata'] else None
      )
