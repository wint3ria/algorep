class MPI_process:  # TODO: Singleton
    '''
    Implements the send, receive and log function for all the
    kinds of MPI_process.
    '''
    def __init__(self, rank, comm, verbose, appname, clock=0, savelog=False):
        self.rank = rank
        self.verbose = verbose
        self.comm = comm
        self.clock = clock
        self.savelog = savelog
        if self.savelog:
            self.logfile = open(f'process{self.rank}_{appname}.log', 'w')

    # TODO: better src/dst handling using MPI status objects

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

    def log(self, msg, highlight=False):
        msg = 'N{} [clk|{}]: {}'.format(self.rank, self.clock, msg)
        if highlight:
            msg = f'\033[93m{msg}\033[0m'
        self.verbose and print(msg, flush=True)
        if self.savelog:
            self.logfile.write(msg + '\n')
            self.logfile.flush()
