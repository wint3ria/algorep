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
node_size = random.randint(15, 25)


'''
data field meaning:
=> -10 : memory_initialisation
=> 1 : Asking memory
'''


class DummyApp:
    def __init__(self, allocator):
        self.allocator = allocator

    def run(self):
        pass


def main():
    allocator = TreeAllocator(rank, nb_children, comm, node_size, verbose=VERBOSE)
    allocator_thread = Thread(target=allocator.run)
    allocator_thread.start()

    app = DummyApp(allocator)
    app.run()

    if not rank:
        allocator.stop_allocator()

    allocator_thread.join()


if __name__ == "__main__":
    main()
    MPI.Finalize()
