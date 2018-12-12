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


class Allocator:
    def __init__(self, rank, comm, size, verbose=False):
        self.comm = comm
        self.rank = rank
        self.verbose = verbose
        self.variables = {}
        self.clock = 0
        self.local_size = size
        self.stop = False
        self.handlers = {}

    def log(self, msg):
        self.verbose and print('N{} [clk|{}]: {}'.format(self.rank, self.clock, msg), flush=True)

    def _send(self, data, dest, tag):
        data = {'clock': self.clock, 'data': data, 'src': self.rank, 'dst': dest}
        req = self.comm.isend(data, dest=dest, tag=tag)
        req.wait()
        self.clock += 1
        self.log(f"send: {data} on tag {tag}")

    def _receive(self, src, tag):
        req = self.comm.irecv(source=src, tag=tag)
        self.log('waiting for {}'.format(src))
        data = req.wait()
        self.log('done waiting for {}'.format(src))
        self.clock = max(self.clock, data['clock']) + 1
        self.log(f'received: {data} on tag {tag}')
        return data

    def run(self):
        while not self.stop:
            request = self._receive(MPI.ANY_SOURCE, 1)
            handler_id = request['data']['handler']
            self.log(f'Call handler "{handler_id}"')
            if handler_id not in self.handlers:
                raise RuntimeError(f'No available handler for this id {handler_id}')
            self.handlers[handler_id](request['data'])


class TreeAllocator(Allocator):

    def __init__(self, rank, nb_children, comm, size, verbose=False):
        super(TreeAllocator, self).__init__(rank, comm, size, verbose)
        self.nb_children = nb_children
        # use a tree topology
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < comm.Get_size()]
        self.parent = None
        if rank:
            self.parent = (rank - 1) // nb_children
        self.memory_map = None
        if self.children:
            self.memory_map = dict([(x, 0) for x in self.children])
        self.size = size
        self._init_memory()
        self.handlers = {
            'stop': self._stop_handler,
            'stop_request': self._request_stop_handler,
            'allocation_request': self._alloc_handler,
            'notification_handler': self._notification_handler,
        }

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
            if self.variables[vid] is not None:
                return self.variables[vid].value
            else:
                pass
                # TODO: request access to a variable outside of the process
        else:
            self.log(f'ERROR: variable {vid} does not exist')
            self._request_stop_handler({'message': 'ERROR: request access to a non existent variable'})
            return None

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
        self.log(f'End of process {self.rank}, variables:\n{self.variables}')

    def _request_stop_handler(self, data):
        if self.rank:
            self._send({'handler': 'request_stop', 'message': data['message']}, self.parent, 3)
        else:
            self.stop_allocator()

    def _alloc_local(self, request_process):
        if len([x for x in self.variables if x is not None]) < self.local_size:
            var = Variable(request_process, self.rank)
            self.variables[var.id] = var
            self._notify_allocation(var.id)
            return var.id
        return None

    def _alloc_handler(self, data):
        self.log('Alloc handler')
        res = self.dalloc(data['request_process'])
        self._send(res, self.parent, 2)

    def _alloc_children(self, request_process):
        children = filter(lambda c: self.memory_map[c] > 0, self.children)
        for child in children:
            self._send({'handler': 'allocation_request', 'request_process': request_process}, child, 1)
            status = self._receive(child, 2)
            vid = status['data']
            if vid is not None:
                self.memory_map[child] -= 1
                self.size -= 1
                self.log(f'Variable {vid} allocated on distant node')
                self.variables[vid] = None
                return vid
        return

    def _notify_allocation(self, vid):
        if self.parent is not None:
            self._send({'data': {'vid': vid, 'from': self.rank}, 'handler': 'notification_handler'}, self.parent, 1)
        for child in self.children:
            self._send({'data': {'vid': vid, 'from': self.rank}, 'handler': 'notification_handler'}, child, 1)

    def _notification_handler(self, data):
        vid = data['data']['vid']
        self.log(f'Allocation notification with vid {vid}, metadata: {data}')
        self.variables[vid] = None
        if self.parent is not None and self.parent != data['data']['from']:
            self._send({'data': data['data'], 'handler': 'notification_handler'}, self.parent, 1)
        children = [c for c in self.children if c != data['data']['from']]
        for child in children:
            self._send({'handler': 'notification_handler', 'data': data['data']}, child, 1)
