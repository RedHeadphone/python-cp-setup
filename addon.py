from main import itertools, Counter, heappush, heappop, bisect_left, bisect_right


class SegmentTree:
    def __init__(self, arr, func=lambda x, y: x + y, default_value_func=lambda: 0):
        self.n = 1 << (len(arr) - 1).bit_length()
        self.func = func
        self.default_value_func = default_value_func
        self.segmentTree = [self.default_value_func() for _ in range(2 * self.n)]
        self.segmentTree[self.n : self.n + len(arr)] = arr
        for i in range(self.n - 1, 0, -1):
            self.segmentTree[i] = self.func(
                self.segmentTree[2 * i], self.segmentTree[2 * i + 1]
            )

    def query(self, l, r):
        l += self.n
        r += self.n
        resl = self.default_value_func()
        resr = self.default_value_func()
        while l < r:
            if l & 1:
                resl = self.func(resl, self.segmentTree[l])
                l += 1
            l >>= 1
            if r & 1:
                r -= 1
                resr = self.func(self.segmentTree[r], resr)
            r >>= 1
        return self.func(resl, resr)

    def update(self, i, value):
        i += self.n
        self.segmentTree[i] = value
        while i > 1:
            i >>= 1
            self.segmentTree[i] = self.func(
                self.segmentTree[2 * i], self.segmentTree[2 * i + 1]
            )


class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = list(range(n))
        self.count = [1] * n
        self.sets_count = n

    def find(self, x):
        x_copy = x
        while self.parents[x] != x:
            x = self.parents[x]
        while x_copy != x:
            x_copy, self.parents[x_copy] = self.parents[x_copy], x
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            if self.count[x] < self.count[y]:
                x, y = y, x
            self.parents[y] = x
            self.count[x] += self.count[y]
            self.sets_count -= 1


class LazyHeap:
    def __init__(self):
        self.heap = []
        self.count = 0
        self.toremove = Counter()

    def push(self, item):
        heappush(self.heap, item)
        self.count += 1

    def remove(self, item):
        self.toremove[item] += 1
        self.count -= 1

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
        return x


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

    def query(self, l, r, x):
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
        return ret


def bit_count(self, num):
    """
    Returns an array of bit counts for each bit position from 0 to num.
    """
    b = len(self.bin(num))
    bs = [0] * b
    c = 2**b - 1
    for i in range(b):
        c -= 1 << i
        bs[i] = (num & c) // 2
        if num & (1 << i):
            bs[i] += (num % (1 << i)) + 1
    return bs


def principal_of_inclusion_exclusion(n):
    for mask in range(1, 1 << n):
        yield mask, pow(-1, mask.bit_count() + 1)


def mo_s_algorithm(queries, state, add, remove, get):
    """
    :param queries: [(l, r, idx)]
    :param add: (i, state) => void
    :param remove: (i, state) => void
    :param get: state => any
    """
    bsize = 500
    queries.sort(key=lambda q: (q[0] // bsize, q[1] * (-1) ** (q[0] // bsize % 2)))
    cur_l, cur_r = 0, -1
    answers = [0] * len(queries)

    for l, r, idx in queries:
        while cur_l > l:
            cur_l -= 1
            add(cur_l, state)
        while cur_r < r:
            cur_r += 1
            add(cur_r, state)
        while cur_l < l:
            remove(cur_l, state)
            cur_l += 1
        while cur_r > r:
            remove(cur_r, state)
            cur_r -= 1
        answers[idx] = get(state)

    return answers


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
        """
        :param merge_func: (node, node) => node
        :param lazy_update_func: (lazy_node, node) => node
        :param lazy_merge_func: (lazy_node, lazy_node) => lazy_node
        """
        self._op = merge_func
        self._e = default_value_func
        self._mapping = lazy_update_func
        self._composition = lazy_merge_func
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
