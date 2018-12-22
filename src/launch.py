from mpi4py import MPI
import random

from tree_allocator import TreeAllocator
from tests import test_applications


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = True
random.seed(rank)

nb_children = 3
node_size = 25


def main():
    if size < 2:
        raise RuntimeError('No process is assigned to the application')

    partition_comm = comm.Split(rank < size // 2, rank)

    for application_ctor in test_applications:
        if rank == 0:
            print(f'Test application: {application_ctor.__name__}', flush=True)

        if rank < size // 2:
            process = TreeAllocator(rank, nb_children, comm, node_size, size // 2, verbose=VERBOSE)
        else:
            allocator_rank = random.randint(0, size // 2 - 1)
            process = application_ctor(rank, allocator_rank, comm, verbose=VERBOSE, app_com=partition_comm)
        comm.barrier()
        process.run()

        if rank >= size // 2:
            partition_comm.barrier()

        if rank == size // 2:
            process.log('Call termination procedure on allocator')
            process._send({'handler': '_request_stop_handler'}, process.allocator_rank, 1)


if __name__ == "__main__":
    main()
    MPI.Finalize()
