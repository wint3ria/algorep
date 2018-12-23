from application import Application
from storage import Variable, Array
import random


class QuickSort(Application):
    def quicksort(self, vid, size):
        if size < 2:
            print('Sort an array of at least 2 elements please.')
            return
        self.sort_partition(vid, 0, size-1)

    def sort_partition(self, vid, start, end):
        pivot = self.read(vid, index=end)
        border = start
        if start < end:
            border_value = self.read(vid, index=border)
            for i in range(start, end+1):
                xi = self.read(vid, index=i)
                if xi <= pivot:
                    self.write(vid, border_value, i)
                    self.write(vid, xi, border)
                    if i != end:
                        border += 1
                        border_value = self.read(vid, index=border)
            self.sort_partition(vid, start, border-1)
            self.sort_partition(vid, border+1, end)

    def run(self):
        size=50
        random.seed()
        arr = random.sample(range(size * 3), size) # random array of len = size
        if self.app_com.Get_rank() == 0:
            vid = self.allocate(size=size)
            if vid is not None:
                # init
                arr_len = min(size, len(arr))
                for i in range(arr_len):
                    self.write(vid, arr[i], i)

                # sorting
                before_sort = arr
                self.quicksort(vid, size)
                after_sort = []
                for i in range(arr_len):
                    after_sort.append(self.read(vid, index=i))
                print(f'--- size={size}\nBefore_sort = {before_sort}')
                print(f'\nAfter_sort = {after_sort}\n---')
            else:
                print('Array too big for us.')
                print('Create more process with `mpiexec -n 8 python launch --quicksort`')
