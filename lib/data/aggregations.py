class AggFn:
    @staticmethod
    def flat_map(value, map_block):
        values = set()
        for vs in value:
            for v in vs:
                values.add(map_block(v))
        return list(values)

    @classmethod
    def flatmap(clazz, map_block = lambda it: it):
        return lambda value: clazz.flat_map(value, map_block)

    @staticmethod
    def lower(): return lambda it: it.lower()   

    @classmethod
    def flatmap_join(clazz, sep=', ', map_block = lambda it: it):
        return lambda value: sep.join(clazz.flat_map(value, map_block))