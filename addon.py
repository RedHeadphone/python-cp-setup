import itertools
from collections import Counter
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right


class MergeSortTree:
    def __init__(self, arr):
        self.log = len(arr).bit_length()
        if len(arr) == 1 << (self.log - 1):
            self.log -= 1
        self.n = 1 << self.log
        self.tree = [[0] * self.n for _ in range(self.log + 1)]
        self.csum = [[0] * self.n for _ in range(self.log + 1)]

        for depth in range(self.log + 1):
            for i in range(len(arr)):
                self.tree[depth][i] = arr[i]

        for depth in range(self.log + 1):
            w = 1 << depth
            for i in range(self.n // w):
                self.tree[depth][i * w : (i + 1) * w] = sorted(
                    self.tree[depth][i * w : (i + 1) * w]
                )
                self.csum[depth][i * w : (i + 1) * w] = list(
                    itertools.accumulate(self.tree[depth][i * w : (i + 1) * w])
                )

    def query(self, l, r, x, less_than_equal=True):
        separate = []
        ret = 0
        for depth in range(self.log + 1):
            size = 1 << depth
            if r - l < size:
                break
            if l % (2 * size) == 0:
                continue
            separate.append((l, l + size))
            l += size

        for depth in range(self.log, -1, -1):
            size = 1 << depth
            if r - l >= size:
                separate.append((l, l + size))
                l += size

        for l, r in separate:
            depth = (r - l).bit_length() - 1
            if less_than_equal:  # <=x
                ok, ng = l - 1, r
                while ng - ok > 1:
                    m = (ng + ok) // 2
                    if self.tree[depth][m] <= x:
                        ok = m
                    else:
                        ng = m
                index = ng - 1
                if index >= l:
                    ret += self.csum[depth][index]
            else:  # >=x
                ok, ng = r, l - 1
                while ok - ng > 1:
                    m = (ok + ng) // 2
                    if self.tree[depth][m] >= x:
                        ok = m
                    else:
                        ng = m
                index = ok
                if index < r:
                    if index == l:
                        ret += self.csum[depth][r - 1]
                    else:
                        ret += self.csum[depth][r - 1] - self.csum[depth][index - 1]
        return ret


def principal_of_inclusion_exclusion(arr, func):
    n = len(arr)
    for j in range(n):
        multiplier = (-1) ** j
        for comb in itertools.combinations(arr, j + 1):
            yield (func(comb), multiplier)


class LazyHeap:
    def __init__(self):
        self.heap = []
        self.count = 0
        self.sum = 0
        self.toremove = Counter()

    def push(self, item):
        heappush(self.heap, item)
        self.count += 1
        self.sum += item

    def remove(self, item):
        self.toremove[item] += 1
        self.count -= 1
        self.sum -= item

    def top(self):
        x = self.heap[0]
        while self.toremove[x] > 0:
            self.toremove[x] -= 1
            heappop(self.heap)
            x = self.heap[0]
        return x

    def pop(self):
        x = self.top()
        heappop(self.heap)
        self.count -= 1
        self.sum -= x
        return x


class LazySegmentTree:
    def __init__(
        self,
        arr,
        merge_func,
        default_value_func,
        lazy_update_func,
        lazy_merge_func,
        default_lazy_func,
    ):
        self._op = merge_func
        self._e = default_value_func
        self._mapping = lazy_update_func  # (lazy_node, node) => node
        self._composition = lazy_merge_func  # (lazy_node, lazy_node) => lazy_node
        self._id = default_lazy_func
        self._n = len(arr)
        self._log = (self._n - 1).bit_length()
        self._size = 1 << self._log
        self._d = [self._e()] * (2 * self._size)
        self._lz = [self._id()] * self._size
        for i in range(self._n):
            self._d[self._size + i] = arr[i]
        for i in range(self._size - 1, 0, -1):
            self._update(i)

    def _update(self, k):
        self._d[k] = self._op(self._d[2 * k], self._d[2 * k + 1])

    def _all_apply(self, k, f):
        self._d[k] = self._mapping(f, self._d[k])
        if k < self._size:
            self._lz[k] = self._composition(f, self._lz[k])

    def _push(self, k):
        self._all_apply(2 * k, self._lz[k])
        self._all_apply(2 * k + 1, self._lz[k])
        self._lz[k] = self._id()

    def set(self, p, x):
        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        self._d[p] = x
        for i in range(1, self._log + 1):
            self._update(p >> i)

    def get(self, p):
        p += self._size
        for i in range(self._log, 0, -1):
            self._push(p >> i)
        return self._d[p]

    def range_query(self, left, right):
        if left == right:
            return self._e()
        left += self._size
        right += self._size
        for i in range(self._log, 0, -1):
            if ((left >> i) << i) != left:
                self._push(left >> i)
            if ((right >> i) << i) != right:
                self._push(right >> i)
        sml = self._e()
        smr = self._e()
        while left < right:
            if left & 1:
                sml = self._op(sml, self._d[left])
                left += 1
            if right & 1:
                right -= 1
                smr = self._op(self._d[right], smr)
            left >>= 1
            right >>= 1
        return self._op(sml, smr)

    def range_update(self, left, right, f):
        if left == right:
            return
        left += self._size
        right += self._size
        for i in range(self._log, 0, -1):
            if ((left >> i) << i) != left:
                self._push(left >> i)
            if ((right >> i) << i) != right:
                self._push((right - 1) >> i)
        l2 = left
        r2 = right
        while left < right:
            if left & 1:
                self._all_apply(left, f)
                left += 1
            if right & 1:
                right -= 1
                self._all_apply(right, f)
            left >>= 1
            right >>= 1
        left = l2
        right = r2
        for i in range(1, self._log + 1):
            if ((left >> i) << i) != left:
                self._update(left >> i)
            if ((right >> i) << i) != right:
                self._update((right - 1) >> i)


class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self._ft_build()
        self.size = 0
        for item in iterable:
            self.add(item)

    def _ft_build(self):
        bit = self.bit = list(self.micro_size)
        bit_size = self.bit_size = len(bit)
        for i in range(bit_size):
            j = i | (i + 1)
            if j < bit_size:
                bit[j] += bit[i]

    def _ft_update(self, idx, x):
        while idx < self.bit_size:
            self.bit[idx] += x
            idx |= idx + 1

    def _ft_get(self, end):
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def _ft_find_kth(self, k):
        idx = -1
        for d in reversed(range(self.bit_size.bit_length())):
            right_idx = idx + (1 << d)
            if right_idx < self.bit_size and self.bit[right_idx] <= k:
                idx = right_idx
                k -= self.bit[idx]
        return idx + 1, k

    def add(self, x):
        i = bisect_left(self.macro, x)
        j = bisect_right(self.micros[i], x)
        self.micros[i].insert(j, x)
        self.size += 1
        self.micro_size[i] += 1
        self._ft_update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i : i + 1] = (
                self.micros[i][: self.block_size >> 1],
                self.micros[i][self.block_size >> 1 :],
            )
            self.micro_size[i : i + 1] = self.block_size >> 1, self.block_size >> 1
            self._ft_build()
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self._ft_update(i, -1)
        return self.micros[i].pop(j)

    def remove(self, x):
        c = self.count(x)
        ind = self.bisect_left(x)
        for _ in range(c):
            self.pop(ind)

    def __getitem__(self, k):
        i, j = self._find_kth(k)
        return self.micros[i][j]

    def count(self, x):
        return self.bisect_right(x) - self.bisect_left(x)

    def __contains__(self, x):
        return self.count(x) > 0

    def bisect_left(self, x):
        i = bisect_left(self.macro, x)
        return self._ft_get(i) + bisect_left(self.micros[i], x)

    def bisect_right(self, x):
        i = bisect_right(self.macro, x)
        return self._ft_get(i) + bisect_right(self.micros[i], x)

    def _find_kth(self, k):
        return self._ft_find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))
