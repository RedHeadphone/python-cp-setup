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

def execute_once():
    pass

def solve(case=None):
    pass


NUMBER_OF_TEST_CASES = 2
DEBUG_ENABLED = 0
BOOLEAN_RETURN = 0
MOD = 10**9 + 7
TRUE_MAPPING, FALSE_MAPPING = "YES", "NO"

###############################################################################


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


class RollingHash:
    MOD1 = 10**9 + 7
    MOD2 = 10**9 + 9

    def __init__(self, string="", base=256, func=ord):
        self.base = base
        self.func = func
        self.MOD1 = RollingHash.MOD1
        self.MOD2 = RollingHash.MOD2
        self.hash1 = 0
        self.hash2 = 0
        self.length = 0
        for i in string:
            self.right_add(i)

    @staticmethod
    def precompute_powers(base, length):
        global pow
        b1, b2 = [1] * length, [1] * length
        r1, r2 = pow(base, RollingHash.MOD1 - 2, RollingHash.MOD1), pow(base, RollingHash.MOD2 - 2, RollingHash.MOD2)
        for i in range(1, length):
            b1[i] = (b1[i - 1] * base) % RollingHash.MOD1
            b2[i] = (b2[i - 1] * base) % RollingHash.MOD2
        
        old_pow = pow
        
        def new_pow(b,e,mod):
            if b==base:
                if mod==RollingHash.MOD1:
                    if 0<=e<length:
                        return b1[e]
                    elif e==RollingHash.MOD1-2:
                        return r1
                elif mod==RollingHash.MOD2:
                    if 0<=e<length:
                        return b2[e]
                    elif e==RollingHash.MOD2-2:
                        return r2
            return old_pow(b,e,mod)
        
        pow = new_pow

    def right_add(self, char):
        self.hash1 = (self.hash1 * self.base + self.func(char)) % self.MOD1
        self.hash2 = (self.hash2 * self.base + self.func(char)) % self.MOD2
        self.length += 1

    def right_remove(self, char):
        self.hash1 = ( (self.hash1 - self.func(char)) * pow(self.base, self.MOD1 - 2, self.MOD1) ) % self.MOD1
        self.hash2 = ( (self.hash2 - self.func(char)) * pow(self.base, self.MOD2 - 2, self.MOD2) ) % self.MOD2
        self.length -= 1

    def update(self, i, value, old_value):
        self.hash1 = (self.hash1 - pow(self.base, self.length - i - 1, self.MOD1) * (self.func(old_value) - self.func(value))) % self.MOD1
        self.hash2 = (self.hash2 - pow(self.base, self.length - i - 1, self.MOD2) * (self.func(old_value) - self.func(value))) % self.MOD2

    def left_add(self, char):
        self.hash1 = (self.hash1 + self.func(char) * pow(self.base, self.length, self.MOD1)) % self.MOD1
        self.hash2 = (self.hash2 + self.func(char) * pow(self.base, self.length, self.MOD2)) % self.MOD2
        self.length += 1

    def left_remove(self, char):
        self.hash1 = (self.hash1 - self.func(char) * pow(self.base, (self.length -1), self.MOD1)) % self.MOD1
        self.hash2 = (self.hash2 - self.func(char) * pow(self.base, (self.length -1), self.MOD2)) % self.MOD2
        self.length -= 1

    def get_hash(self):
        return self.hash1 * self.MOD2 + self.hash2
    

class RangeHash:
    def __init__(self, string="", base=256, func=ord, reverse=False):
        rolling_hash_pre = RollingHash(base=base, func=func)
        self.MOD1 = rolling_hash_pre.MOD1
        self.MOD2 = rolling_hash_pre.MOD2

        n = len(string)
        self.base = base
        self.prefix_hash1 = [0] * (n + 1)
        self.prefix_hash2 = [0] * (n + 1)
        if reverse:
            rolling_hash_suf = RollingHash(base=base, func=func)
            self.suffix_hash1 = [0] * (n + 1)
            self.suffix_hash2 = [0] * (n + 1)
        
        for i in range(n):
            rolling_hash_pre.right_add(string[i])
            self.prefix_hash1[i + 1] = rolling_hash_pre.hash1
            self.prefix_hash2[i + 1] = rolling_hash_pre.hash2
            if reverse:
                rolling_hash_suf.right_add(string[n-1-i])
                self.suffix_hash1[n - 1 - i] = rolling_hash_suf.hash1
                self.suffix_hash2[n - 1 - i] = rolling_hash_suf.hash2

    def get_hash(self, l, r):
        hash1 = (self.prefix_hash1[r] - self.prefix_hash1[l] * pow(self.base, r-l, self.MOD1)) % self.MOD1
        hash2 = (self.prefix_hash2[r] - self.prefix_hash2[l] * pow(self.base, r-l, self.MOD2)) % self.MOD2
        return hash1 * self.MOD2 + hash2

    def get_rev_hash(self, l, r):
        hash1 = (self.suffix_hash1[l] - self.suffix_hash1[r] * pow(self.base, r-l, self.MOD1)) % self.MOD1
        hash2 = (self.suffix_hash2[l] - self.suffix_hash2[r] * pow(self.base, r-l, self.MOD2)) % self.MOD2
        return hash1 * self.MOD2 + hash2


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

execute_once()

for t in range(NUMBER_OF_TEST_CASES if NUMBER_OF_TEST_CASES <= 1 else int(input())):
    if BOOLEAN_RETURN:
        print(TRUE_MAPPING if solve(t + 1) else FALSE_MAPPING)
    else:
        solve(t + 1)
