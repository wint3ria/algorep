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
        self.clock = 0

    def allocate(self, size):
        if self.allocated + size < node_size:
            var = Variable(size)
            self.variables.add(var)
            # notify the allocation to other processes
            return var
        else:
            pass

    def send(self, data, dest, tag):
        req = comm.isend(data, dest=dest, tag=tag)
        req.wait()
        self.clock += 1
        print(rank, "send: ", data, dest)

    def receive(self, src, tag):
        req = comm.irecv(source = src, tag = tag)
        data = req.wait()
        self.clock = max(self.clock, data[0])
        print(rank, 'received: ', data[1], src)
        return data[1]


    def init_memory(self):
        # if node have a parent, wait for its memory request
        if self.parent != None:
            parent_data = self.receive(self.parent, 11)
            # memory request is represented by the int '-10' in the data field
            if parent_data != -10:
                print('incorrect data for memory initialisation')

        # if the node have children, send memory request and wait for answer
        if self.children:
            for child in self.children:
                data = (self.clock, -10)
                self.send(data, child, 11)

            for child in self.children:
                data = self.receive(child, 11)
                self.allocated += data

        # if node have again a parent, send all the info back to him
        if self.parent != None:
            if parent_data == -10:
                data = (self.clock, self.allocated)
                self.send(data, self.parent, 11)

        # print to verify the allocated variable per node
        if self.children:
            print(rank, self.allocated)


class DummyApp:
    def __init__(self, allocator):
        random.seed(rank)
        self.allocator = allocator

    def run(self):
        pass


def main():
    allocator = Allocator()
    allocator.init_memory()

    app = DummyApp(allocator)
    app.run()


if __name__ == "__main__":
    main()

MPI.Finalize()
