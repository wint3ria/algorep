from mpi4py import MPI
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
        self._send({'handler': 'dmalloc'}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def free(self, vid):
        self._send({
                'handler': 'dfree',
                'send_back': self.rank,
                'vid': vid,
            }, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)

    def write(self, vid, value):
        self._send({
                'handler': 'dwrite',
                'send_back': self.rank,
                'vid': vid,
                'value': value,
            }, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)


class SimpleAllocTest(Application):
    def run(self):
        while True:
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id).value}')


class MultipleReadTest(Application):
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
                    msg = f'Invalid read operation on app {self.rank} with vid {vid}. Read method returned: {var}'
                    msg += f'\nAllocator rank: {self.allocator_rank}'
                    raise RuntimeError(msg)
                self.log(var)

class SimpleFreeTest(Application):
    def run(self):
        free_tries = 2
        while free_tries:
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request free on variable {var_id}')
            freed = self.free(var_id)['data']
            self.log(f'Freed: {freed}')
            if freed:
                free_tries -= 1

class SimpleWriteTest(Application):
    def run(self):
        while True:
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id)}')
            value = 67
            self.log(f'Request write on variable {var_id} with value {value}')
            wrote = self.write(var_id, value)['data']
            self.log(f'Wrote: {wrote}')
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id)}')
            if wrote:
                break


test_applications = [
    # SimpleAllocTest,
    SimpleWriteTest,
    # SimpleFreeTest,
    # MultipleReadTest,
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
        allocator_rank = random.randint(0, size // 2 - 1)
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
