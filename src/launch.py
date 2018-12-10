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


'''
data field meaning:
=> -10 : memory_initialisation
=> 1 : Asking memory
'''


class DummyApp:
    def __init__(self, allocator):
        self.allocator = allocator

    def run(self):
        var_id = True
        while var_id and not rank:
            self.allocator.log(f'Request allocation')
            var_id = self.allocator.dalloc()
            self.allocator.log(f'Allocation done, got id {var_id}')
        self.allocator.read_variable((0, 0, 0))
        self.allocator.read_variable((0, 0, 1))
        self.allocator.read_variable((0, 0, 1))
        self.allocator.read_variable((12, 42))
        self.allocator.read_variable((0, 1, 0))


def main():
    allocator = TreeAllocator(rank, nb_children, comm, node_size, verbose=VERBOSE)
    allocator_thread = Thread(target=allocator.run)
    allocator_thread.start()

    app = DummyApp(allocator)
    app.run()

    if not rank:
        sleep(3)
        allocator.stop_allocator()

    allocator_thread.join()


if __name__ == "__main__":
    main()
    MPI.Finalize()
