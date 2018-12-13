from time import sleep

from mpi4py import MPI
from threading import Thread
import random


from allocator import TreeAllocator


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = True
random.seed(rank)

nb_children = 3
node_size = 2


class StressSimpleAlloc:
    def __init__(self, allocator):
        self.allocator = allocator

    def run(self):
        self.allocator.log(f'Request allocation')
        var_id = self.allocator.dmalloc()
        self.allocator.log(f'Allocation done, got id {var_id}')
        self.allocator.log(f'Request read on variable {var_id}')
        self.allocator.log(f'Got this value: {self.allocator.read_variable(var_id)}')


def main():
    allocator = TreeAllocator(rank, nb_children, comm, node_size, verbose=VERBOSE)
    allocator_thread = Thread(target=allocator.run)
    allocator_thread.start()

    app = StressSimpleAlloc(allocator)
    app.run()

    if not rank:
        sleep(3)
        allocator.stop_allocator()

    allocator_thread.join()


if __name__ == "__main__":
    main()
    MPI.Finalize()
