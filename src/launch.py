from time import sleep

from mpi4py import MPI
from threading import Thread
import random


from allocator import TreeAllocator


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = True
random.seed(rank)

nb_children = 3
node_size = 2


class StressSimpleAlloc:
    def __init__(self, rank, allocator_rank, comm, verbose=VERBOSE):
        self.rank = rank
        self.allocator_rank = allocator_rank
        self.verbose = VERBOSE
        self.comm = comm
        self.clock = 0

    def _send(self, data, dest, tag):
        data = {'clock': self.clock, 'data': data, 'src': self.rank, 'dst': dest}
        req = self.comm.isend(data, dest=dest, tag=tag)
        self.clock += 1
        req.wait()
        self.log(f"send: {data} on tag {tag}")

    def _receive(self, src, tag):
        req = self.comm.irecv(source=src, tag=tag)
        self.log('waiting for {}'.format(src))
        data = req.wait()
        self.log('done waiting for {}'.format(src))
        self.clock = max(self.clock, data['clock']) + 1
        self.log(f'received: {data} on tag {tag}')
        return data

    def log(self, msg):
        msg = 'App {} [clk|{}]: {}'.format(self.rank, self.clock, msg)
        self.verbose and print(msg, flush=True)

    def read(self, vid):
        self._send({'handler': 'read_request_handler', 'vid': vid}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 5)

    def allocate(self):
        self._send({'handler': 'allocation_request'}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 2)

    def run(self):
        self.log(f'Request allocation')
        var_id = self.allocate()['data']
        self.log(f'Allocation done, got id {var_id}')
        self.log(f'Request read on variable {var_id}')
        self.log(f'Got this value: {self.read(var_id)}')

def main():
    if rank < size / 2:
        allocator = TreeAllocator(rank, nb_children, comm, node_size, size / 2, verbose=VERBOSE)
        allocator.run()
    else:
        app = StressSimpleAlloc(rank, 0, comm, verbose=VERBOSE)
        app.run()


if __name__ == "__main__":
    main()
    MPI.Finalize()
