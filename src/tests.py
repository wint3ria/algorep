from application import Application
from storage import Variable, Array


test_applications = []


def register_app(app):
    global test_applications
    test_applications.append(app)
    return app


@register_app
class SimpleAlloc(Application):
    def run(self):
        for _ in range(5):
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id).value}')


@register_app
class MultipleRead(Application):
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


@register_app
class SimpleFree(Application):
    def run(self):
        free_tries = 2
        while free_tries:
            self.log(f'Request allocation')
            var_id = self.allocate()
            self.log(f'Allocation done, got id {var_id}')
            if var_id is None:
                break
            self.log(f'Request free on variable {var_id}')
            freed = self.free(var_id)
            self.log(f'Freed: {freed}')
            if freed:
                free_tries -= 1


@register_app
class SimpleWrite(Application):
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
            wrote = self.write(var_id, value)
            self.log(f'Wrote: {wrote}')
            self.log(f'Request read on variable {var_id}')
            self.log(f'Got this value: {self.read(var_id)}')
            if wrote:
                break


@register_app
class SimpleArray(Application):
    def run(self):
        if self.app_com.Get_rank() == 0:
            self.log('Array allocation test')
            vid = self.allocate(size=4)
            self.log(f'Received {vid}')
            if vid is None:
                self.log('Not enough memory!')
                return
            var = self.read(vid, index=3)
            self.log(f'allocated var: {var}')


@register_app
class SimpleArrayWrite(Application):
    def run(self):
        if self.app_com.Get_rank() == 0:
            self.log('Array write test')
            vid = self.allocate(size=4)
            self.log(f'Received {vid}')
            if vid is None:
                self.log('Not enough memory!')
                return
            for i in range(4):
                self.log(f'Writing value {4 - i} at index {i}')
                self.write(vid, 4 - i, i)
            for i in range(4):
                var = self.read(vid, index=i)
                self.log(f'Read value {var} at index {i}')


@register_app
class BigArrayAlloc(Application):
    def run(self):
        if self.app_com.Get_rank() == 0:
            self.log('Big array allocation test')
            vid = self.allocate(size=6)
            if vid is None:
                self.log('Could not allocate the "big" array')
                return
            else:
                self.log('Successfully allocated a "big" array')
            for i in range(4):
                self.log(f'READING......... {i}', True)
                self.log(f'from={self.rank}', True)
                self.log(self.read(vid, index=i), True)
            return vid


@register_app
class BigArrayWrite(BigArrayAlloc):
    def run(self):
        if self.app_com.Get_rank() == 0:
            vid = super().run()
            if vid is not None:
                i = 1
                self.log(f'Writing value {-i} at index {i}')
                self.write(vid, -i, i)

                var = self.read(vid, index=i)
                self.log(f'\n\nRead value {var} at index {i}\n\n')
                for i in range(6):
                    self.log(f'Writing value {- i} at index {i}')
                    self.write(vid, -i, i)
                tab = []
                for i in range(6):
                    var = self.read(vid, index=i)
                    tab.append(var)
                    self.log(f'\n\nRead value {var} at index {i}\n\n')
                self.log(f'\n\n\n end: {tab}\n')
