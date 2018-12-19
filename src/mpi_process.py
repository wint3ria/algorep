from mpi4py import MPI

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
