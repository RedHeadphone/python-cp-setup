import random
import math
from collections import defaultdict, Counter, deque
from functools import lru_cache
import bisect
import heapq
import sys

input = sys.stdin.readline
# sys.setrecursionlimit(10**6) 

class SegmentTree:

    def __init__(self, nums):
        self._size = len(nums)
        self._tree = [0] * (4 * self._size)
        self.build(nums)

    def build(self, a, v=1, lo=0, hi=None):
        if hi is None:
            hi = self._size - 1
            
        if lo == hi:
            self._tree[v] = a[lo]
        else:
            mi = (lo + hi) // 2
            self.build(a, 2 * v, lo, mi)
            self.build(a, 2 * v + 1, mi + 1, hi)
            self._tree[v] = self._tree[2 * v] + self._tree[2 * v + 1]
        
    def update(self, pos, val, v=1, lo=0, hi=None):
        if hi is None:
            hi = self._size - 1
            
        if lo == hi:
            self._tree[v] = val
        else:
            mi = (lo + hi) // 2
            if pos <= mi:
                self.update(pos, val, 2 * v, lo, mi)
            else:
                self.update(pos, val, 2 * v + 1, mi + 1, hi)
            
            self._tree[v] = self._tree[2 * v] + self._tree[2 * v + 1]
        
    def query(self, l, h, v=1, lo=0, hi=None):
        if hi is None:
            hi = self._size - 1
            
        if l > h:
            return 0
        elif l == lo and h == hi:
            return self._tree[v]
        else:
            mi = (lo + hi) // 2
            return self.query(l, min(mi, h), 2 * v, lo, mi) +\
                   self.query(max(mi + 1, l), h, 2 * v + 1, mi + 1, hi)

class UnionFind:
    def __init__(self, n):
        self.n = n
        self.parents = list(range(n))
    def find(self, x):
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]
    def union(self, x, y):
        x = self.find(x)
        y = self.find(y)
        if x != y:
            self.parents[x] = y

dire = [0,1,0,-1,0]

def is_prime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
    return True

def case(t):
    print("Case #{}:".format(t), end=" ")

RANDOM = random.randrange(2**62)
def Wrapper(x):
  return x ^ RANDOM

###############################################################################

def solve():
    pass

###############################################################################

for t in range(int(input())):
    # case(t+1)
    solve()