class MPI_process:  # TODO: Singleton
    def __init__(self, rank, comm, verbose, appname, clock=0):
        self.rank = rank
        self.verbose = verbose
        self.comm = comm
        self.clock = clock
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

    def log(self, msg):
        if self.verbose:
            msg = 'N{} [clk|{}]: {}'.format(self.rank, self.clock, msg)
            print(msg, flush=True)
            self.logfile.write(msg + '\n')
            self.logfile.flush()
