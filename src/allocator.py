from mpi4py import MPI


'''
MPI tags

0: init memory
'''


class Variable:
    num = 0

    def __init__(self, size, rank):
        self.value = [None] * size
        self.id = (rank, Variable.num)
        Variable.num += 1

    def __hash__(self):
        return hash(self.id)


class TreeAllocator:
    def __init__(self, rank, nb_children, comm, size, verbose=False):
        self.comm = comm
        self.rank = rank
        self.verbose = verbose
        # use a tree topology
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < comm.Get_size()]
        self.parent = None
        if rank:
            self.parent = (rank - 1) // nb_children
        self.variables = set()
        self.allocated = 0
        self.local_allocated = 0
        self.clock = 0
        self.memory_map = None
        if self.children:
            self.memory_map = dict([(x, 0) for x in self.children])
        self.size = size
        self.local_size = size

    def allocate(self, size):
        if self.local_allocated + size < self.local_size: # local allocation only on this node
            var = Variable(size, self.rank)
            self.variables.add(var)
            self.allocated += size
            # TODO: notify function
            return
        if self.allocated + size < self.size:
            pass

    def log(self, msg):
        self.verbose and print('N{} [clk|{}]: {}'.format(self.rank, self.clock, msg), flush=True)

    def run(self):
        while True:
            data = self.receive(MPI.ANY_SOURCE, MPI.ANY_TAG)
            print(data)

    def send(self, data, dest, tag, val_id=None, size=None):
        data = {'clock': self.clock, 'data': data, 'src': self.rank}
        if val_id:
            data['id'] = val_id
        if size:
            data['size'] = size
        req = self.comm.isend(data, dest=dest, tag=tag)
        req.wait()
        self.clock += 1
        self.log("send: {}".format(data))

    def receive(self, src, tag):
        req = self.comm.irecv(source=src, tag=tag)
        self.log('waiting for {}'.format(src))
        data = req.wait()
        self.log('done waiting for {}'.format(src))
        self.clock = max(self.clock, data['clock']) + 1
        self.log('received: {}'.format(data))
        return data

    def init_memory(self):
        self.log('call init mem')
        for child in self.children:
            child_size = self.receive(child, 0)['data']
            self.memory_map[child] = child_size
            self.size += child_size
        if self.parent is not None:
            self.send(self.size, self.parent, 0)
        self.log('end of init_memory. subtree size: {}; memory map: {}'.format(self.size, self.memory_map))
