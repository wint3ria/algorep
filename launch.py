from mpi4py import MPI
import random


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nb_children = 3
node_size = 20


class Variable:
    def __init__(self, size):
        self.value = [None] * size


class Allocator:
    def __init__(self):
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < size]
        self.parent = None
        if rank:
            self.parent = (rank - 1) // nb_children
        self.variables = set()
        self.allocated = 0

    def allocate(self, size):
        if self.allocated + size < node_size:
            var = Variable(size)
            self.variables.add(var)
            # notify the allocation to other processes
            return var
        else:
            pass

    def init_memory(self):
        pass


class DummyApp:
    def __init__(self, allocator):
        random.seed(rank)
        self.allocator = allocator

    def run(self):
        pass


def main():
    allocator = Allocator()
    app = DummyApp(allocator)
    app.run()


if __name__ == "__main__":
    main()

MPI.Finalize()
