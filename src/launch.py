from time import sleep

from mpi4py import MPI
from threading import Thread
import random


from allocator import TreeAllocator, MPI_process


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = True
random.seed(rank)

nb_children = 3
node_size = 2


class StressSimpleAlloc(MPI_process):
    def __init__(self, rank, allocator_rank, comm, verbose):
        super(StressSimpleAlloc, self).__init__(rank, comm, size, verbose)
        self.allocator_rank = allocator_rank

    def read(self, vid):
        self._send({'handler': 'read_request_handler', 'vid': vid}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 5)

    def allocate(self):
        self._send({'handler': 'allocation_request'}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 2)

    def run(self):
        self.log(f'Request allocation')
        var_id = self.allocate()['data']
        self.log(f'Allocation done, got id {var_id}')
        self.log(f'Request read on variable {var_id}')
        self.log(f'Got this value: {self.read(var_id)}')

def main():
    if rank < size / 2:
        allocator = TreeAllocator(rank, nb_children, comm, node_size, size / 2, verbose=VERBOSE)
        allocator.run()
    else:
        app = StressSimpleAlloc(rank, 0, comm, verbose=VERBOSE)
        app.run()


if __name__ == "__main__":
    main()
    MPI.Finalize()
