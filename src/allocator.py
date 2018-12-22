from functools import wraps

from mpi4py import MPI
from mpi_process import MPI_process
import traceback


handlers = []
translation_table = {}


def register_handler(handler):
    name = handler.__name__
    translation_table[name] = len(handlers)
    handler.id = len(handlers)
    handlers.append(handler)
    return handler


def public_handler(handler):

    @wraps(handler)
    def wrapper(*args, **kwargs):
        if 'master' not in args[1]['data']:
            args[1]['data']['master'] = args[1]['dst']
            args[1]['data']['caller'] = args[1]['src']
        return handler(*args, **kwargs)

    return wrapper


instantiation_id = 0


class Allocator(MPI_process):
    def __init__(self, rank, comm, size, verbose=False):
        global instantiation_id
        super(Allocator, self).__init__(rank, comm, verbose, f'Allocator{instantiation_id}')
        instantiation_id += 1
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
                self.log(f'exception: {traceback.format_exc()}\nOn allocator: {self}')
                self.stop = True
                self.comm.Abort(1)

    def __repr__(self):
        r = f'{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}'
        return f'{r} variables={self.variables}, local size={self.local_size}, stop={self.stop}'
