from application import Application
from storage import Variable, Array


test_applications = []


def register_app(app):
    global test_applications
    test_applications.append(app)


class SimpleAllocTest(Application):
    def run(self):
        for _ in range(5):
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id).value}')
register_app(SimpleAllocTest)


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
                if type(var) != Variable and type(var) != Array:
                    msg = f'Invalid read operation on app {self.rank} with vid {vid}. Read method returned: {var}'
                    msg += f'\nAllocator rank: {self.allocator_rank}'
                    raise RuntimeError(msg)
                self.log(var)
register_app(MultipleReadTest)


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
register_app(SimpleFreeTest)


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
register_app(SimpleWriteTest)

class SimpleArrayTest1(Application):
    def allocate(self):
        self._send({'handler': 'dmalloc', 'size': 4}, self.allocator_rank, 1)
        return self._receive(self.allocator_rank, 10)['data']

    def run(self):
        if self.app_com.Get_rank() == 0:
            self.log('Array allocation test')
            vid = self.allocate()
            self.log(f'Received {vid}')
            var = self.read(vid, index=3)
            self.log(f'allocated var: {var}')
