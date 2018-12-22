from allocator import Allocator, register_handler, public_handler
from storage import Variable, Array


class TreeAllocator(Allocator):  # TODO: docstrings
    def __init__(self, rank, nb_children, comm, size, tree_size, verbose=False):
        super(TreeAllocator, self).__init__(rank, comm, size, verbose)
        self.tree_size = tree_size
        self.nb_children = nb_children
        # use a tree topology
        self.children = [x for x in range(rank * nb_children + 1, (rank + 1) * nb_children + 1) if x < tree_size]
        self.parent = None
        if rank:
            self.parent = (rank - 1) // nb_children

    def response_handler(self, metadata, return_value_id='response'):
        data = metadata['data']
        master = data['master']
        caller = data['caller']
        if self.rank == master:
            self._send(data[return_value_id], caller, 10)
            return
        if master in self.children:
            self._send(data, master, 1)
            return
        is_ancestor, path = _is_ancestor(self.rank, master, self.nb_children, self.tree_size)
        if not is_ancestor:
            self._send(data, self.parent, 1)
            return
        self._send(data, path[-2], 1)

    @register_handler
    @public_handler
    def dfree_response_handler(self, metadata):
        data = metadata['data']
        if data['vid'] in self.variables:
            self.variables.pop(data['vid'], None)
            self.local_size += 1
            data['response'] = True
            metadata['data'] = data
        self.response_handler(metadata)

    # TODO: handle arrays free

    @register_handler
    @public_handler
    def dfree(self, metadata):
        self.search_tree(metadata, self.dfree_response_handler)

    @register_handler
    @public_handler
    def dwrite_response_handler(self, metadata):
        data = metadata['data']
        vid = data['vid']
        if vid in self.variables:
            if type(self.variables[vid]) == Variable:  # Simple variable assignment
                # TODO: clock
                self.variables[vid].value = data['value']
            else:  # Array value assignment
                index = data['index']
                if index < self.variables[vid].size:  # The current array contains the index
                    # TODO: clock
                    self.variables[vid].value[index] = data['value']
                else:  # Search the next array in the linked list
                    data['index'] -= self.variables[vid].size
                    data['vid'] = self.variables[vid].next
                    self.dwrite(metadata)
                    return
            metadata['data']['response'] = True
        self.response_handler(metadata)

    '''
    TODO: Add clock management for the writes on the variables
    This is the previously used code, before removing send_back

    if data['variable'].last_write_clock < metadata['clock']:
        if type(data['variable']) == Array:
            data['variable'].value[data['index']] = data['value']
        else:
            data['variable'].value = data['value']
        data['variable'].last_write_clock = metadata['clock']
        self._send(True, dst, 10)
    else:
        self._send(False, dst, 10)
    '''

    @register_handler
    @public_handler
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
        if data['vid'] is None and metadata['src'] in self.children:
            if 'excluded' in data:
                data['excluded'] = data['excluded'] + [metadata['src']]
            else:
                data['excluded'] = [metadata['src']]
        self.response_handler(metadata, 'vid')

    @register_handler
    @public_handler
    def dmalloc(self, metadata):
        data = metadata['data']
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
            arr_size = min(size, self.local_size)
            ctor = lambda req, rank: Array(req, rank, arr_size, next)

        local_alloc_size = min(size, self.local_size)
        child_alloc_size = size - local_alloc_size

        var = None
        if local_alloc_size != 0:
            self.local_size -= local_alloc_size
            var = ctor(data['caller'], self.rank)
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
                self._send(data, child, 1)
                return

        if self.parent is not None:
            self._send(data, self.parent, 1)
            return
        data['vid'] = None
        data['handler'] = 'dmalloc_response_handler'
        metadata['data'] = data
        self.dmalloc_response_handler(metadata)

    @register_handler
    def read_response_handler(self, metadata):
        data = metadata['data']
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
                    self.read_variable(metadata)
                    return
        self.response_handler(metadata, 'variable')

    @register_handler
    @public_handler
    def read_variable(self, metadata):
        self.search_tree(metadata, self.read_response_handler)

    def search_tree(self, metadata, response_handler):
        data = metadata['data']
        vid = data['vid']
        # self.log(f'search_tree: data={data}', True)
        owner = vid[1]

        if vid in self.variables:
            metadata['data'] = data
            response_handler(metadata)
            return

        # src = metadata['src']
        children = self.children  # [child for child in self.children if child != src]

        '''
        TODO: bugfix: Sometimes, a child possessing the variable asks for its parent.
        This causes the following test to fail if we filter the children as commented above.
        As it is more a matter of optimisation, I removed the filtering for the moment.
        '''

        if owner in children or owner == self.parent:
            self.log('Child or parent owns the variable')
            data['handler'] = response_handler.__name__
            self._send(data, owner, 1)
            return

        is_ancestor, path = _is_ancestor(self.rank, owner, self.nb_children, self.tree_size)
        if is_ancestor:
            self.log(path)
            self.log(owner)
            self.log(children)
            self._send(data, path[-2], 1)  # TODO: Fix out of range
            return
        self._send(data, self.parent, 1)


def _is_ancestor(a, n, k, tree_size, l=list()):
    if n >= tree_size:
        return False, l
    if n == 0:
        return False, l
    an = (n - 1) // k
    l.append(an)
    if an == a:
        return True, l
    if an == 0:
        return False, None
    return _is_ancestor(a, an, k, tree_size, l)
