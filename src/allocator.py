from mpi4py import MPI
from mpi_process import MPI_process
import traceback


handlers = []
translation_table = {}


def register_handler(handler, name=None):
    name = name or handler.__name__
    translation_table[name] = len(handlers)
    handler.id = len(handlers)
    handlers.append(handler)
    return handler


class Allocator(MPI_process):
    def __init__(self, rank, comm, size, verbose=False):
        super(Allocator, self).__init__(rank, comm, verbose)
        self.variables = {}
        self.local_size = size
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                request = self._receive(MPI.ANY_SOURCE, 1)
                handler_name = request['data']['handler']
                self.log(f'Call handler "{handler_name}"')
                if handler_name not in translation_table:
                    raise RuntimeError(f'No available handler for this id {handler_name}')
                handlers[translation_table[handler_name]](self, request)

            except Exception as e:
                self.log(f'exception: {traceback.format_exc()}')
                self.stop = True
