import traceback
from mpi4py import MPI
import random
import argparse

from tree_allocator import TreeAllocator
from tests import test_applications


parser = argparse.ArgumentParser(description='Launch a distributed allocator and some unit tests'
                                             'or a distributed quicksort implemention')
parser.add_argument('--node_size', help="Number of variable an allocator can possess", default=2, type=int)
parser.add_argument('--nb_children', help="Number of children for each node", default=3, type=int)
parser.add_argument('--quicksort', help="Launch a distributed quicksort implementation instead of unit tests",
                    default=False, action="store_true")
parser.add_argument('--verbose', action="store_true", help="Enable verbose mode", default=False)
args = parser.parse_args()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
VERBOSE = args.verbose
random.seed(rank)
nb_children = args.nb_children
node_size = args.node_size


def tests():
    if size < 2:
        raise RuntimeError('No process is assigned to the application')

    partition_comm = comm.Split(rank < size // 2, rank)

    for application_ctor in test_applications:
        try:
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

            status = "SUCCESS"
        except Exception:
            print(traceback.format_exc(), flush=True)
            status = "FAIL"

        if rank == 0:
            print(f'Test application: {application_ctor.__name__}; Status: {status}', flush=True)


def main():
    # TODO: Implement quicksort
    pass


if __name__ == "__main__":
    if args.quicksort:
        main()
    else:
        tests()
    MPI.Finalize()
