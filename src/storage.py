num = 0


class Storage:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)


# TODO: Better access/write methods


class Variable(Storage):
    def __init__(self, request_process, rank, vid=None):
        if vid is None:
            global num
            super().__init__((request_process, rank, num))
            num += 1
        else:
            super().__init__(vid)
        self.value = None
        self.last_write_clock = -1

    def __repr__(self):
        default_repr = f'{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}'
        return f'<{default_repr}, val={self.value}>'


class Array(Variable):
    def __init__(self, request_process, rank, size, next):
        super().__init__(request_process, rank)
        self.size = size
        self.value = [None] * self.size
        self.next = next

    def __repr__(self):
        r = super().__repr__()
        return f'{r}, size={self.size}, next={self.next}'
