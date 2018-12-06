from mpi4py import MPI
import random


from allocator import TreeAllocator


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nb_children = 3
node_size = 20
VERBOSE = True


'''
data field meaning:
=> -10 : memory_initialisation
=> 1 : Asking memory
'''


class DummyApp:
    def __init__(self, allocator):
        random.seed(rank)
        self.allocator = allocator

    def run(self):
        pass


def main():
    allocator = TreeAllocator(rank, nb_children, comm, node_size, verbose=VERBOSE)
    allocator.init_memory()
    '''if rank == 1:
        allocator.allocate(10)
        allocator.allocate(19)
    if rank == 0:
        for i in range(2): #waiting for 2 messages, but should be a while true i think
            allocator.listen()
    '''
    app = DummyApp(allocator)
    app.run()


if __name__ == "__main__":
    main()
    MPI.Finalize()
