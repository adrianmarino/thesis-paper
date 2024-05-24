def iterable_object(cls):
    def __iter__(self):
        self.index = 0
        return self
    cls.__iter__ = __iter__

    def __len__(self): return len(self._elements())
    cls.__len__ = __len__

    def __next__(self):
        if self.index < len(self):
            value = self._elements()[self.index]
            self.index += 1
            return value
        else:
            raise StopIteration()
    cls.__next__ = __next__

    return cls