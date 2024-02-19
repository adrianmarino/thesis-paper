from repository.chroma import WhereMetadataBuilder


class ItemSimQuery:
    def __init__(self):
        self.__content = None
        self.__limit   = 5
        self.__user_id = None
        self.__rating  = 0
        self.__seen    = False
        self.__where_medata = WhereMetadataBuilder()
    
    def user_id_eq(self, value):
        self.__user_id = value
        return self

    def is_seen(self, value):
        self.__seen = value
        return self

    def rating_gte(self, value):
        if value is not None:
            value = int(value)
            if value > 0:
                self.__rating = value
        return self

    def contains(self, value):
        if value is not None and len(value) > 0:
            self.__content = value
        return self

    def limit_eq(self, value):
        if value is not None:
            value = int(value)
            if value > 0:
                self.__limit = value
        return self

    def release_gte(self, value):
        self.__where_medata.gte('release', value)
        return self

    def id_in(self, values, negate=False):
        self.__where_medata.is_in('id', values, negate=negate)
        return self

    @property
    def rating(self): return self.__rating

    @property
    def seen(self): return self.__seen

    @property
    def user_id(self): return self.__user_id

    @property
    def limit(self): return self.__limit

    @property
    def content(self): return self.__content

    @property
    def where(self): return self.__where_medata.build()
