from mpi4py import MPI


'''
MPI tags

0: init memory
1: ask execution of a procedure
'''

num = 0


class Storage:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)


class Variable(Storage):
    def __init__(self, size, rank, vid=None):
        if vid:
            global num
            super().__init__((rank, num))
            num += 1
        else:
            super().__init__(vid)
        self.value = [None] * size


class Bucket(Storage):
    def __init__(self, allocator, size):
        global num
        super().__init__((allocator.rank, num))
        num += 1
        local_allocation = allocator.local_size - allocator.local_allocated
        subsize = size / allocator.nb_children + 1
        local_allocation = min(local_allocation, subsize)
        size -= local_allocation

        for i, child in enumerate(allocator.children):
            subsize = size / (allocator.nb_children - i)
            sub_allocation = min(subsize, allocator.memory_map[child])
            allocator.send({'size': sub_allocation, 'id': self.id, 'handler': 'allocation_request'}, child, 1)


class TreeAllocator:
    def __init__(self, rank, nb_children, comm, size, verbose=False):
        self.nb_children = nb_children
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
        self.stop = False
        self.handlers = {
            'allocation_request': self.allocation_handler,
            'stop_request': self.stop_handler,
        }
        self.init_memory()

    def allocation_handler(self, data):
        self.allocate(data['size'], vid=data['id'])

    def allocate(self, size, vid=None):
        if self.local_allocated + size < self.local_size:  # local allocation only on this node
            var = Variable(size, self.rank, vid=vid)
            self.variables.add(var)  # TODO: handle child allocation fail case
            self.allocated += size
            # TODO: notify function
            return var
        if self.allocated + size < self.size:  # allocation scattered on the children
            bucket = Bucket(self, size)
            self.variables.add(bucket)
            # TODO: notify function
            return bucket
        # Last case, we should ask our parent
        self.send({'size': size, 'id': vid, 'handler': 'allocation_request'}, self.parent, 1)

    def log(self, msg):
        self.verbose and print('N{} [clk|{}]: {}'.format(self.rank, self.clock, msg), flush=True)

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

    def run(self):
        while not self.stop:
            request = self.receive(MPI.ANY_SOURCE, 1)
            handler_id = request['data']['handler']
            self.handlers[handler_id](request['data'])

    def stop_handler(self, data):
        self.stop = True
        for child in self.children:
            self.send({'handler': 'stop_request'}, child, 1)

    def stop_allocator(self):
        if self.rank != 0:
            raise RuntimeError('Must be called by the root only')
        self.send({'handler': 'stop_request'}, 0, 1)
