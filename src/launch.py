from time import sleep

from mpi4py import MPI
from threading import Thread
import random


from allocator import TreeAllocator, MPI_process, Variable


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = True
random.seed(rank)

nb_children = 3
node_size = 2


class Application(MPI_process):
    def __init__(self, rank, allocator_rank, comm, verbose, app_com=None):
        super(Application, self).__init__(rank, comm, size, verbose)
        self.allocator_rank = allocator_rank
        if app_com:
            self.app_com = app_com

    def read(self, vid):
        self._send({
                'handler': 'read_variable',
                'send_back': self.rank,
                'vid': vid,
            },
            self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def allocate(self):
        self._send({'handler': 'allocation_request'}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 2)['data']


class SimpleAllocTest1(Application):
    def run(self):
        self.log(f'Request allocation')
        var_id = self.allocate()
        self.log(f'Allocation done, got id {var_id}')
        self.log(f'Request read on variable {var_id}')
        self.log(f'Got this value: {self.read(var_id)}')
        self.read((self.allocator_rank, self.allocator_rank, 0))


class MultipleReadTest1(Application):
    def run(self):
        self.log('Request Allocation')
        vid = self.allocate()
        self.log(f'Allocation id: {vid}')
        recv_vid_buf = self.app_com.allgather([vid])
        self.log(recv_vid_buf)
        for vid_buf in recv_vid_buf:
            for vid in filter(lambda x: x is not None, vid_buf):
                var = self.read(vid)
                if type(var) != Variable:
                    raise RuntimeError(f'Invalid read operation on app {self.rank} with vid {vid}.\
                    Read method returned: {var}')
                self.log(var)


test_applications = [
    SimpleAllocTest1,
    MultipleReadTest1,
]


def main():
    if size < 2:
        raise RuntimeError('No process is assigned to the application')

    if rank < size // 2:
        process = TreeAllocator(rank, nb_children, comm, node_size, size // 2, verbose=VERBOSE)

    # Make the application processes wait te end of the initialisation of the allocator
    partition_comm = comm.Split(rank < size // 2, rank)
    comm.barrier()

    if rank < size // 2:
        process.run()

    # Run the different applications defined for testing
    if rank >= size // 2:
        allocator_rank = random.randint(0, size // 2)
        for application_ctor in test_applications:
            process = application_ctor(rank, allocator_rank, comm, verbose=VERBOSE, app_com=partition_comm)
            process.run()
            partition_comm.barrier()

    # Call termination function
    if rank == size // 2:
        process.log('Call termination procedure on allocator')
        process._send({'handler': 'stop_request'}, process.allocator_rank, 1)


if __name__ == "__main__":
    main()
    MPI.Finalize()
