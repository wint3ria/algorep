from mpi4py import MPI


'''
MPI tags

0: init memory
1: ask execution of a procedure
2: allocation result
3: request stop procedure
4: notify a local allocation
'''

num = 0


class Storage:
    def __init__(self, id):
        self.id = id

    def __hash__(self):
        return hash(self.id)


class Variable(Storage):
    def __init__(self, request_process, rank, vid=None):
        if vid is None:
            global num
            super().__init__((request_process, rank, num))
            num += 1
        else:
            super().__init__(vid)
        self.value = None


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
        self.variables = {}
        self.clock = 0
        self.memory_map = None
        if self.children:
            self.memory_map = dict([(x, 0) for x in self.children])
        self.size = size
        self.local_size = size
        self.stop = False
        self.handlers = {
            'stop': self._stop_handler,
            'stop_request': self._request_stop_handler,
            'allocation_request': self._alloc_handler,
        }
        self._init_memory()

    def log(self, msg):
        self.verbose and print('N{} [clk|{}]: {}'.format(self.rank, self.clock, msg), flush=True)

    def run(self):
        while not self.stop:
            request = self._receive(MPI.ANY_SOURCE, 1)
            handler_id = request['data']['handler']
            self.handlers[handler_id](request['data'])

    def stop_allocator(self):  # TODO: atomic function
        if self.rank != 0:
            raise RuntimeError('Must be called by the root only')
        if not self.stop:
            self._send({'handler': 'stop'}, 0, 1)

    def dalloc(self, request_process=None):
        self.clock += 1
        res = self._alloc_local(request_process or self.rank)
        if not res:
            res = self._alloc_children(request_process or self.rank)
        # TODO: ask parent for allocation
        return res

    def read_variable(self, vid):
        self.log(f'Read access to variable {vid}')
        if vid in self.variables:
            return dict(self.variables)[vid]
        if vid not in self.variables:
            self.log(f'ERROR: variable {vid} does not exist')
            self._request_stop_handler({'message': 'ERROR: request access to a non existant variable'})
            return None
        # TODO: request access to a variable outside of the process

    def _send(self, data, dest, tag):
        data = {'clock': self.clock, 'data': data, 'src': self.rank, 'dst': dest}
        req = self.comm.isend(data, dest=dest, tag=tag)
        req.wait()
        self.clock += 1
        self.log("send: {}".format(data))

    def _receive(self, src, tag):
        req = self.comm.irecv(source=src, tag=tag)
        self.log('waiting for {}'.format(src))
        data = req.wait()
        self.log('done waiting for {}'.format(src))
        self.clock = max(self.clock, data['clock']) + 1
        self.log('received: {}'.format(data))
        return data

    def _init_memory(self):
        self.log('call init memory')
        for child in self.children:
            child_size = self._receive(child, 0)['data']
            self.memory_map[child] = child_size
            self.size += child_size
        if self.parent is not None:
            self._send(self.size, self.parent, 0)
        self.log('end of init_memory. subtree size: {}; memory map: {}'.format(self.size, self.memory_map))

    def _stop_handler(self, data):
        self.stop = True
        for child in self.children:
            self._send({'handler': 'stop'}, child, 1)

    def _request_stop_handler(self, data):
        if self.rank:
            self._send({'handler': 'request_stop', 'message': data['message']}, self.parent, 3)
        else:
            self.stop_allocator()

    def _alloc_local(self, request_process):
        if len(self.variables) < self.local_size:
            var = Variable(request_process, self.rank)
            self.variables[var.id] = var
            # TODO: notify allocation
            return var.id
        return None

    def _notify_allocation(self, vid):
        self._send({'data': vid, 'handler': 'notification_handler'}, self.parent, 4)
        for child in self.children:
            self._send({'data': vid, 'handler': 'notification_handler'}, child, 4)

    def _notification_handler(self, data):
        pass

    def _alloc_handler(self, data):
        self.log('Alloc handler')
        res = self.dalloc(data['request_process'])
        self._send(res, self.parent, 2)

    def _alloc_children(self, request_process):
        children = filter(lambda c: self.memory_map[c] > 0, self.children)
        for child in children:
            self._send({'handler': 'allocation_request', 'request_process': request_process}, child, 1)
            status = self._receive(child, 2)
            if status['data'] is not None:
                self.memory_map[child] -= 1
                return status['data']
        return None
