from allocator import Allocator, register_handler
from storage import Variable, Array


class TreeAllocator(Allocator):  # TODO: docstrings
    def __init__(self, rank, nb_children, comm, size, tree_size, verbose=False):
        super(TreeAllocator, self).__init__(rank, comm, size, verbose)
        self.nb_children = nb_children
        # use a tree topology
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < tree_size]
        self.parent = None
        if rank:
            self.parent = (rank - 1) // nb_children

    @register_handler
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

    # TODO: handle arrays free

    @register_handler
    def dfree(self, metadata):
        self.search_tree(metadata, self.dfree_response_handler)

    @register_handler
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
            if 'index' not in data:
                self.variables[data['vid']].value = data['value']
            else:
                self.variables[data['vid']].value[data['index']] = data['value']
            self.variables[data['vid']].last_write_clock = metadata['clock']
            self._send(True, dst, 10)
        else:
            self.log(f'Didn\'t write {vid} because the clock is too late')
            self._send(False, dst, 10)

    @register_handler
    def dwrite(self, metadata):
        self.search_tree(metadata, self.dwrite_response_handler)

    @register_handler
    def _stop_handler(self, metadata):
        self.stop = True
        for child in self.children:
            self._send({'handler': '_stop_handler'}, child, 1)
        self.log(f'End of process {self.rank}, variables:\n{self.variables}')

    @register_handler
    def _request_stop_handler(self, metadata):
        data = metadata['data']
        if self.rank:
            new_data = {'handler': '_request_stop_handler'}
            if 'message' in data:
                new_data['message'] = data['message']
            self._send(new_data, self.parent, 1)
        else:
            self._stop_handler(None)

    @register_handler
    def dmalloc_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if data['vid'] is None and metadata['src'] in self.children:
            if 'excluded' in data:
                data['excluded'] = data['excluded'] + [metadata['src']]
            else:
                data['excluded'] = [metadata['src']]
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        self._send(data['vid'], dst, 10)

    @register_handler
    def dmalloc(self, metadata):
        data = metadata['data']
        if 'send_back' not in data:
            data['send_back'] = [metadata['src']]
        if metadata['src'] in self.children:
            if 'excluded' in data:
                data['excluded'] = data['excluded'] + [metadata['src']]
            else:
                data['excluded'] = [metadata['src']]
        next = None
        excluded = []
        if 'excluded' in data:
            excluded = data['excluded']

        if 'prev' in data:
            next = data['prev']
        if 'size' not in data or data['size'] == 1:
            ctor = Variable
            size = 1
        else:
            size = data['size']
            ctor = lambda req, rank: Array(req, rank, min(size, self.local_size), next)

        local_alloc_size = min(size, self.local_size)
        child_alloc_size = size - local_alloc_size

        var = None
        if local_alloc_size != 0:
            self.local_size -= local_alloc_size
            var = ctor(data['send_back'][0], self.rank)
            self.variables[var.id] = var
            data['vid'] = var.id
            if child_alloc_size == 0:
                data['handler'] = 'dmalloc_response_handler'
                metadata['data'] = data
                self.dmalloc_response_handler(metadata)
                return

        data['size'] = child_alloc_size
        data['prev'] = None
        if var is not None:
            data['prev'] = var.id

        if len(self.children) != 0:
            children = [x for x in self.children if x not in excluded]
            if len(children) != 0:
                child = min(children)
                data['send_back'].append(self.rank)
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

    @register_handler
    def read_response_handler(self, metadata):
        data = metadata['data']
        dst = data['send_back'].pop()
        if 'variable' not in data:
            if type(self.variables[data['vid']]) == Variable:
                data['variable'] = self.variables[data['vid']]
            else:
                index = data['index']
                tab = self.variables[data['vid']]
                if index < tab.size:
                    data['variable'] = tab.value[index]
                else:
                    data['index'] -= tab.size
                    data['vid'] = tab.next
                    metadata['data'] = self.read_variable(metadata)
                    return
        if len(data['send_back']):
            self._send(data, dst, 1)
            return
        self._send(data['variable'], dst, 10)

    @register_handler
    def read_variable(self, metadata):
        self.search_tree(metadata, self.read_response_handler)

    def search_tree(self, metadata, response_handler):
        data = metadata['data']
        vid = data['vid']
        owner = vid[1]

        send_back = data['send_back']
        if type(send_back) == int:
            data['src'] = send_back
            send_back = [send_back]
        data['send_back'] = send_back

        if vid in self.variables:
            metadata['data'] = data
            response_handler(metadata)
            return

        src = data['src']
        children = [child for child in self.children if child != src]
        data['src'] = self.rank
        send_back.append(self.rank)

        if owner in children or owner == self.parent:
            self.log('Child or parent owns the variable')
            data['handler'] = response_handler.__name__
            self._send(data, owner, 1)
            return

        is_ancestor, path = _is_ancestor(self.rank, owner, self.nb_children)
        if is_ancestor:
            self._send(data, path[-2], 1)  # TODO: Fix out of range
            return
        self._send(data, self.parent, 1)


def _is_ancestor(a, n, k, l=list()):
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
    return _is_ancestor(a, an, k, l)
