from mpi_process import MPI_process


class Application(MPI_process):
    def __init__(self, rank, allocator_rank, comm, verbose, app_com=None, log=False):
        super(Application, self).__init__(rank, comm, verbose, self.__class__.__name__, savelog=log)
        self.allocator_rank = allocator_rank
        if app_com:
            self.app_com = app_com

    def read(self, vid, index=None):
        data = {
            'handler': 'read_variable',
            'vid': vid,
        }
        if index is not None:
            data['index'] = index
        self._send(data, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def allocate(self, size=1):
        self._send({'handler': 'dmalloc', 'size': size}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def free(self, vid):
        self._send({
                'handler': 'dfree',
                'vid': vid,
            }, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def write(self, vid, value, index=None):
        data = {
                'handler': 'dwrite',
                'vid': vid,
                'value': value,
        }
        if index is not None:
            data['index'] = index
        self._send(data, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']
