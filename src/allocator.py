from mpi4py import MPI
from storage import Variable, Array
from mpi_process import MPI_process
import traceback


class Allocator(MPI_process):
    def __init__(self, rank, comm, size, tree_size, verbose=False, allow_notifications=False):
        super(Allocator, self).__init__(rank, comm, verbose)
        self.variables = {}
        self.local_size = size
        self.tree_size = tree_size
        self.stop = False
        self.handlers = {}
        self.allow_notifications = allow_notifications

    def run(self):
        while not self.stop:
            try:
                request = self._receive(MPI.ANY_SOURCE, 1)
                handler_id = request['data']['handler']
                self.log(f'Call handler "{handler_id}"')
                if handler_id not in self.handlers:
                    raise RuntimeError(f'No available handler for this id {handler_id}')
                self.handlers[handler_id](request)
            except Exception as e:
                self.log(f'exception: {traceback.format_exc()}')
                self.stop = True
