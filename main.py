import sys, random, math, string, itertools
from copy import deepcopy
from collections import defaultdict, Counter, deque
from heapq import heapify, heappush, heappop
from functools import cache
from bisect import bisect_left, bisect_right
from types import GeneratorType
from typing import *

input = lambda: sys.stdin.readline().strip()


def debug(*x, **y):
    if not DEBUG_ENABLED:
        return
    print(*x, (y if y != {} else "\b"), file=sys.stderr)


###############################################################################


def solve_bruteforce(case=None):
    pass


def solve(case=None):
    pass


NUMBER_OF_TEST_CASES = 2
DEBUG_ENABLED = 0
BOOLEAN_RETURN = 0
MOD = 10**9 + 7
TRUE_MAPPING, FALSE_MAPPING = "YES", "NO"

###############################################################################


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
            if index < l:
                continue
            ret += self.csum[depth][index]
        return ret


def principal_of_inclusion_exclusion(arr, func):
    n = len(arr)
    for j in range(n):
        multiplier = (-1) ** j
        for comb in itertools.combinations(arr, j + 1):
            yield (func(comb), multiplier)


class SegmentTree:
    def __init__(self, arr, func=lambda x, y: x + y, defaultvalue=0):
        self.n = len(arr)
        self.segmentTree = [0] * self.n + arr
        self.func = func
        self.defaultvalue = defaultvalue
        for i in range(self.n - 1, 0, -1):
            self.segmentTree[i] = self.func(
                self.segmentTree[2 * i], self.segmentTree[2 * i + 1]
            )

    def query(self, l, r):
        l += self.n
        r += self.n
        res = self.defaultvalue
        while l < r:
            if l & 1:
                res = self.func(res, self.segmentTree[l])
                l += 1
            l >>= 1
            if r & 1:
                r -= 1
                res = self.func(res, self.segmentTree[r])
            r >>= 1
        return res

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


# Custom HashMap and Set for Python's Anti-hash-table test
class CustomHashMap:
    def __init__(self, type=dict, arg=[]):  # allowed types: dict, defaultdict, Counter
        self.RANDOM = random.randrange(2**62)
        if type != defaultdict:
            self.dict = type([self.wrapper(i) for i in arg])
        else:
            self.dict = type(arg)  # arg should be function

    def wrapper(self, num):
        return num ^ self.RANDOM

    def __setitem__(self, key, value):
        self.dict[self.wrapper(key)] = value

    def __getitem__(self, key):
        return self.dict[self.wrapper(key)]

    def __contains__(self, key):
        return self.wrapper(key) in self.dict

    def __iter__(self):
        return iter(self.wrapper(i) for i in self.dict)

    def keys(self):
        return [self.wrapper(i) for i in self.dict]

    def values(self):
        return [i for i in self.dict.values()]

    def items(self):
        return [(self.wrapper(i), j) for i, j in self.dict.items()]

    def __repr__(self):
        return repr({self.wrapper(i): j for i, j in self.dict.items()})


class CustomSet:
    def __init__(self, arr=[]):
        self.RANDOM = random.randrange(2**62)
        self.set = set([self.wrapper(i) for i in arr])

    def wrapper(self, num):
        return num ^ self.RANDOM

    def add(self, key):
        self.set.add(self.wrapper(key))

    def remove(self, key):
        self.set.remove(self.wrapper(key))

    def __contains__(self, key):
        return self.wrapper(key) in self.set

    def __iter__(self):
        return iter(self.wrapper(i) for i in self.set)

    def __len__(self):
        return len(self.set)

    def __repr__(self):
        return repr({self.wrapper(i) for i in self.set})


class lazyheap:
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


class RollingHash:
    def __init__(self, base=256, string="", func=ord):
        self.base = base
        self.hash = 0
        self.length = 0
        self.func = func
        for i in string:
            self.right_add(i)

    def right_add(self, char):
        self.hash = (self.hash * self.base + self.func(char)) % MOD
        self.length += 1

    def left_add(self, char):
        self.hash = (
            self.hash + self.func(char) * pow(self.base, self.length, MOD)
        ) % MOD
        self.length += 1


# recursion limit fix decorator, change 'return' to 'yield' and add 'yield' before calling the function
def bootstrap(f):
    stack = []

    def wrappedfunc(*args, **kwargs):
        if stack:
            return f(*args, **kwargs)
        else:
            to = f(*args, **kwargs)
            while True:
                if type(to) is GeneratorType:
                    stack.append(to)
                    to = next(to)
                else:
                    stack.pop()
                    if not stack:
                        break
                    to = stack[-1].send(to)
            return to

    return wrappedfunc


def binary_search(left, right, check, start_from_left):
    if start_from_left:
        ans = left
    else:
        ans = right
    while left <= right:
        mid = (left + right) // 2
        if start_from_left:
            if check(mid):
                ans, left = mid, mid + 1
            else:
                right = mid - 1
        else:
            if check(mid):
                ans, right = mid, mid - 1
            else:
                left = mid + 1
    return ans


class BitManipulation:
    def bit_length(self, num):
        return num.bit_length()

    def bin(self, num, size=None):
        bin_string = bin(num)[2:]
        return bin_string if (size == None) else bin_string.zfill(size)

    def bit_sum(self, num):  # returns array of sum of bits of all numbers from 0 to num
        b = len(self.bin(num))
        bs = [0] * b
        c = 2**b - 1
        for i in range(b):
            c -= 1 << i
            bs[i] = (num & c) // 2
            if num & ((1 << i)):
                bs[i] += (num % (1 << i)) + 1
        return bs


class Combinatorics:
    def __init__(self, pre_compute_limit=0):
        self.fact = [1] * pre_compute_limit
        self.invfact = [1] * pre_compute_limit
        for i in range(1, pre_compute_limit):
            self.fact[i] = (self.fact[i - 1] * i) % MOD
            self.invfact[i] = (self.invfact[i - 1] * self.mod_inverse(i)) % MOD

    def mod_inverse(self, a):
        return pow(a, MOD - 2, MOD)

    def ncr(self, n, r):
        if n < r:
            return 0
        return (self.fact[n] * self.invfact[r] * self.invfact[n - r]) % MOD

    def npr(self, n, r):
        if n < r:
            return 0
        return (self.fact[n] * self.invfact[n - r]) % MOD

    def ncr_without_memo(self, n, r):
        if n < r:
            return 0
        num = den = 1
        for i in range(r):
            num = (num * (n - i)) % MOD
            den = (den * (i + 1)) % MOD
        return (num * pow(den, MOD - 2, MOD)) % MOD

    def npr_without_memo(self, n, r):
        if n < r:
            return 0
        num = 1
        for i in range(n, n - r, -1):
            num = (num * i) % MOD
        return num


class Factors:
    def __init__(self, pre_compute_limit=0):
        self.last_prime_factor = [0] * pre_compute_limit
        for i in range(2, pre_compute_limit):
            if self.last_prime_factor[i] > 0:
                continue
            for j in range(i, pre_compute_limit, i):
                self.last_prime_factor[j] = i

    def prime_factors(self, n):
        c = Counter()
        while n > 1:
            c[self.last_prime_factor[n]] += 1
            n //= self.last_prime_factor[n]
        prime_factors = list(c.items())
        prime_factors.sort(key=lambda x: x[0])
        return prime_factors

    def prime_factors_without_memo(self, n):
        c = Counter()
        divisor = 2
        while divisor * divisor <= n:
            while n % divisor == 0:
                c[divisor] += 1
                n //= divisor
            divisor += 1
        if n > 1:
            c[n] += 1
        prime_factors = list(c.items())
        prime_factors.sort(key=lambda x: x[0])
        return prime_factors

    def factors_without_memo(self, n):
        if n == 0:
            return set()
        factors = set()
        divisor = 1
        while divisor * divisor <= n:
            if n % divisor == 0:
                factors.add(divisor)
                factors.add(n // divisor)
            divisor += 1
        return factors

    def is_prime(self, n):
        if n < 2:
            return False
        return self.last_prime_factor[n] == n

    def is_prime_without_memo(self, n):
        if n < 2:
            return False
        divisor = 2
        while divisor * divisor <= n:
            if n % divisor == 0:
                return False
            divisor += 1
        return True


class SortedList:
    block_size = 700

    def __init__(self, iterable=()):
        self.macro = []
        self.micros = [[]]
        self.micro_size = [0]
        self.ft_build()
        self.size = 0
        for item in iterable:
            self.add(item)

    def ft_build(self):
        bit = self.bit = list(self.micro_size)
        bit_size = self.bit_size = len(bit)
        for i in range(bit_size):
            j = i | (i + 1)
            if j < bit_size:
                bit[j] += bit[i]

    def ft_update(self, idx, x):
        while idx < self.bit_size:
            self.bit[idx] += x
            idx |= idx + 1

    def ft_get(self, end):
        x = 0
        while end:
            x += self.bit[end - 1]
            end &= end - 1
        return x

    def ft_find_kth(self, k):
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
        self.ft_update(i, 1)
        if len(self.micros[i]) >= self.block_size:
            self.micros[i : i + 1] = (
                self.micros[i][: self.block_size >> 1],
                self.micros[i][self.block_size >> 1 :],
            )
            self.micro_size[i : i + 1] = self.block_size >> 1, self.block_size >> 1
            self.ft_build()
            self.macro.insert(i, self.micros[i + 1][0])

    def pop(self, k=-1):
        i, j = self._find_kth(k)
        self.size -= 1
        self.micro_size[i] -= 1
        self.ft_update(i, -1)
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
        return self.ft_get(i) + bisect_left(self.micros[i], x)

    def bisect_right(self, x):
        i = bisect_right(self.macro, x)
        return self.ft_get(i) + bisect_right(self.micros[i], x)

    def _find_kth(self, k):
        return self.ft_find_kth(k + self.size if k < 0 else k)

    def __len__(self):
        return self.size

    def __iter__(self):
        return (x for micro in self.micros for x in micro)

    def __repr__(self):
        return str(list(self))


for t in range(NUMBER_OF_TEST_CASES if NUMBER_OF_TEST_CASES <= 1 else int(input())):
    if BOOLEAN_RETURN:
        print(TRUE_MAPPING if solve(t + 1) else FALSE_MAPPING)
    else:
        solve(t + 1)
