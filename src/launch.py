from mpi4py import MPI
import random


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nb_children = 3
node_size = 20
PRINT_LOGS = True
'''
data field meaning:
=> -10 : memory_initialisation
=> 1 : Asking memory


'''
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
        self.memory_map = None
        if self.children:
            self.memory_map = dict([ (x, 0) for x in self.children])

    def allocate(self, size):
        if self.allocated + size < node_size:
            var = Variable(size)
            self.variables.add(var)
            self.allocated += size

            # notify the allocation to other processes
            if self.parent != None:
                data = 'memory_allocation'
                self.send(data, self.parent, 3, size=size)
        else:
            if self.parent != None:
                data = 'request_allocation'
                self.send(data, self.parent, 1, size=size)
            else:
                # deal with not enough memory left
                pass

    def send(self, data, dest, tag, val_id = None, size = None):
        data = {'clock' : self.clock, 'data' : data, 'src' : rank}
        if val_id:
            data['id'] = val_id
        if size:
            data['size'] = size
        req = comm.isend(data, dest=dest, tag=tag)
        req.wait()
        self.clock += 1
        if PRINT_LOGS:
            print("send: ", data)

    def receive(self, src, tag):
        req = comm.irecv(source = src, tag = tag)
        data = req.wait()
        self.clock = max(self.clock, data['clock']) + 1
        if PRINT_LOGS:
            print('received: ', data)
        return data


    def init_memory(self):
        # if node have a parent, wait for its memory request
        if self.parent != None:
            parent_data = self.receive(self.parent, 11)
            # memory request is represented by the int '-10' in the data field
            if parent_data['data'] != 'init_memory':
                print('incorrect data for memory initialisation')

        # if the node have children, send memory request and wait for answer
        if self.children:
            for child in self.children:
                data = 'init_memory'
                self.send(data, child, 11)

            for child in self.children:
                data = self.receive(child, 11)
                self.memory_map[child] = data['data']
                self.allocated += data['data']

        # if node have again a parent, send all the info back to him
        if self.parent != None:
            if parent_data['data'] == 'init_memory':
                data = self.allocated
                self.send(data, self.parent, 11)

        # print to verify the allocated variable per node
        print('end of init_memory for rank: ', rank, '; total allocated is: ', self.allocated, '; memory map is: ', self.memory_map)

    def listen(self):
        data = self.receive(MPI.ANY_SOURCE, MPI.ANY_TAG)
        print('listened: ', data)

        if data['data'] == 'request_allocation':
            self.allocate(data['size'])
        elif data['data'] == 'memory_allocation':
            self.memory_map[data['src']] += data['size']
            print('memory alloc', self.memory_map)

class DummyApp:
    def __init__(self, allocator):
        random.seed(rank)
        self.allocator = allocator

    def run(self):
        pass

def main():
    allocator = Allocator()
    allocator.init_memory()
    if rank == 1:
        allocator.allocate(10)
        allocator.allocate(19)
    if rank == 0:
        for i in range(2): #waiting for 2 messages, but should be a while true i think
            allocator.listen()

    app = DummyApp(allocator)


if __name__ == "__main__":
    main()

MPI.Finalize()
