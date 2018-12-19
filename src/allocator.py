from mpi4py import MPI
import traceback

'''
MPI tags

0: init memory
1: ask execution of a procedure
2: allocation result
3: request stop procedure
4: notify a local allocation
5: read request
10: public interface responses
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
        self.last_write_clock = -1

    def __repr__(self):
        default_repr = f'{self.__class__.__module__}.{self.__class__.__name__} at {hex(id(self))}'
        return f'<{default_repr}, val={self.value}>'


class MPI_process:
    def __init__(self, rank, comm, verbose, clock=0):
        self.rank = rank
        self.verbose = verbose
        self.comm = comm
        self.clock = clock
        self.logfile = open(f'process{self.rank}.log', 'w')

    def _send(self, data, dest, tag):
        data = {'clock': self.clock, 'data': data, 'src': self.rank, 'dst': dest}
        self.comm.isend(data, dest=dest, tag=tag)
        self.clock += 1
        self.log(f"send: {data} on tag {tag}")

    def _receive(self, src, tag):
        data = self.comm.recv(source=src, tag=tag)
        self.log('waiting for {}'.format(src))
        self.log('done waiting for {}'.format(src))
        self.clock = max(self.clock, data['clock']) + 1
        self.log(f'received: {data} on tag {tag}')
        return data

    def log(self, msg):
        msg = 'N{} [clk|{}]: {}'.format(self.rank, self.clock, msg)
        self.verbose and print(msg, flush=True)
        self.logfile.write(msg + '\n')
        self.logfile.flush()


class Allocator(MPI_process):

    def __init__(self, rank, comm, size, tree_size, verbose=False, allow_notifications=False):
        super(Allocator, self).__init__(rank, comm, verbose)
        self.variables = {}
        self.local_size = size
        self.tree_size = tree_size
        self.stop = False
        self.handlers = {}
        self.allow_notifications = allow_notifications

    def run(self):
        while not self.stop:
            try:
                request = self._receive(MPI.ANY_SOURCE, 1)
                handler_id = request['data']['handler']
                self.log(f'Call handler "{handler_id}"')
                if handler_id not in self.handlers:
                    raise RuntimeError(f'No available handler for this id {handler_id}')
                self.handlers[handler_id](request)
            except Exception as e:
                self.log(f'exception: {traceback.format_exc()}')
                self.stop = True


class TreeAllocator(Allocator):
    def __init__(self, rank, nb_children, comm, size, tree_size, verbose=False):
        super(TreeAllocator, self).__init__(rank, comm, size, tree_size, verbose)
        self.nb_children = nb_children
        # use a tree topology
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < self.tree_size]
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
            #'allocation_request': self._alloc_handler,
            #'notification_handler': self._notification_handler,
            'dmalloc': self.dmalloc,
            'dmalloc_response_handler': self.dmalloc_response_handler,
            'read_variable': self.read_variable,
            'read_response_handler': self.read_response_handler,
            'dfree': self.dfree,
            'dfree_response_handler': self.dfree_response_handler,
            'dwrite': self.dwrite,
            'dwrite_response_handler': self.dwrite_response_handler,
        }

    def read_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if 'variable' not in data:
            data['variable'] = self.variables[data['vid']]
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        self._send(data['variable'], dst, 10)

    def read_variable(self, metadata):
        self.dsearch(metadata, self.read_response_handler, 'read_variable')

    def dfree_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if 'variable' not in data:
            data['variable'] = self.variables[data['vid']]
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        self.variables.pop(data['vid'], None)
        self.local_size += 1
        self._send(True, dst, 10)

    def dfree(self, metadata):
        self.dsearch(metadata, self.dfree_response_handler, 'dfree')

    def dwrite_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if 'variable' not in data:
            data['variable'] = self.variables[data['vid']]
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        # metadata['clock'] is the source client's clock
        if self.variables[data['vid']].last_write_clock < metadata['clock']:
            self.variables[data['vid']].value = data['value']
            self.variables[data['vid']].last_write_clock = metadata['clock']
            self._send(True, dst, 10)
        else:
            self.log(f'Didn\'t write {vid} because the clock is too late')
            self._send(False, dst, 10)

    def dwrite(self, metadata, direct_addressing=False):
        self.dsearch(metadata, self.dwrite_response_handler, 'dwrite')

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
            new_data = {'handler': 'stop_request'}
            if 'message' in data:
                new_data['message'] = data['message']
            self._send(new_data, self.parent, 1)
        else:
            self._stop_handler(None)


    def dmalloc_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if data['vid'] is None and metadata['src'] in self.children:
            self.memory_map[metadata['src']] = 0
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        self._send(data['vid'], dst, 10)


    def dmalloc(self, metadata):
        data = metadata['data']
        if 'send_back' not in data:
            data['send_back'] = [metadata['src']]

        if metadata['src'] in self.children:
            self.memory_map[metadata['src']] = 0

        if self.local_size > 0:
            self.local_size -= 1
            var = Variable(data['send_back'][0], self.rank)
            self.variables[var.id] = var
            data['vid'] = var.id
            data['handler'] = 'dmalloc_response_handler'
            metadata['data'] = data
            self.dmalloc_response_handler(metadata)
            return

        if self.memory_map is not None:
            children = [x for x in self.children if self.memory_map[x] > 0]
            if len(children) != 0:
                child = min(children)
                data['send_back'].append(self.rank)
                self.memory_map[child] -= 1
                self._send(data, child, 1)
                return

        if self.parent is not None:
            data['send_back'].append(self.rank)
            self._send(data, self.parent, 1)
            return

        data['vid'] = None
        data['handler'] = 'dmalloc_response_handler'
        metadata['data'] = data
        self.dmalloc_response_handler(metadata)

    def dsearch(self, metadata, handler_to_call, caller_name, direct_addressing=False):
        data = metadata['data']
        vid = data['vid']
        owner = vid[1]

        send_back = data['send_back']
        if type(send_back) == int:
            self.log(f'First API call for {caller_name} {vid} from {send_back}')
            data['src'] = send_back
            send_back = [send_back]
        data['send_back'] = send_back

        if direct_addressing:  # Doesn't work, use direct_addressing=False
            metadata['data'] = data
            send_back.append(owner)
            handler_to_call(metadata)
            return

        if vid in self.variables:
            metadata['data'] = data
            handler_to_call(metadata)
            return

        src = data['src']
        children = [child for child in self.children if child != src]
        data['src'] = self.rank
        send_back.append(self.rank)

        if owner in children or owner == self.parent:
            self.log('Child or parent owns the variable')
            data['handler'] = handler_to_call.__name__
            self._send(data, owner, 1)
            return

        is_ancestor, path = self._is_ancestor(self.rank, owner, self.nb_children)
        if is_ancestor:
            self._send(data, path[-2], 1)
            return
        self._send(data, self.parent, 1)










    '''
    def dmalloc(self, request_process=None):
        self.clock += 1
        res = self._alloc_local(request_process or self.rank)
        if not res:
            res = self._alloc_children(request_process or self.rank)
        if not res and self.parent is not None:
            res = self._alloc_parent(request_process or self.rank)
        return res

    def _alloc_local(self, request_process):
        if len([x for x in self.variables if self.variables[x] is not None]) < self.local_size:
            var = Variable(request_process, self.rank)
            self.variables[var.id] = var
            if self.allow_notifications:
                self._notify_allocation(var.id)
            return var.id
        return None

    def _alloc_handler(self, data):
        self.log('Alloc handler')

        request_process = None
        if 'request_process' in data:
            request_process = data['request_process']

        res = self.dmalloc(request_process)
        if self.parent:
            self._send(res, self.parent, 2)
        else:
            self._send(res, data['src'], 2)

    def _alloc_children(self, request_process):
        if request_process in self.children:
            self.log('setting children {} memory to 0. memory map: {}'.format(request_process, self.memory_map))
            self.memory_map[request_process] = 0
        children = filter(lambda c: self.memory_map[c] > 0, self.children)
        for child in children:
            self._send({'handler': 'allocation_request', 'request_process': request_process}, child, 1)

            # TODO: Fix deadlock

            status = self._receive(child, 2)
            vid = status['data']
            if vid is not None:
                self.memory_map[child] -= 1
                self.size -= 1
                self.log(f'Variable {vid} allocated on distant node')
                return vid
        return


    def _alloc_parent(self, request_process):
        self._send({'handler': 'allocation_request', 'request_process': request_process}, self.parent, 1)

        # TODO: Fix deadlock

        status = self._receive(self.parent, 2)
        vid = status['data']
        if vid is not None:
            self.size -= 1
            self.log(f'Variable {vid} allocated on distant node')
            return vid
        return
    '''





















    def _is_ancestor(self, a, n, k, l=list()):
        if n == 0:
            return False, l
        if a == 0:
            return True, l
        an = (n - 1) // k
        l.append(an)
        if an == a:
            return True, l
        if an == 0:
            return False, None
        return self._is_ancestor(a, an, k, l)

    def _notify_allocation(self, vid):
        if self.parent is not None:
            self._send({'data': {'vid': vid, 'from': self.rank}, 'handler': 'notification_handler'}, self.parent, 1)
        for child in self.children:
            self._send({'data': {'vid': vid, 'from': self.rank}, 'handler': 'notification_handler'}, child, 1)

    def _notification_handler(self, data):
        src = data['src']
        data = data['data']
        vid = data['data']['vid']
        self.log(f'Allocation notification with vid {vid}, metadata: {data}')
        self.variables[vid] = None
        if self.parent is not None and self.parent != src:
            self._send({'data': data['data'], 'handler': 'notification_handler'}, self.parent, 1)
        children = [c for c in self.children if c != src]
        for child in children:
            self._send({'handler': 'notification_handler', 'data': data['data']}, child, 1)
