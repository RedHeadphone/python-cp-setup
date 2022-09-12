import random
import math
from collections import defaultdict, Counter, deque, OrderedDict
from functools import lru_cache
import bisect
import heapq
import sys

input = lambda : sys.stdin.readline().strip()
# sys.setrecursionlimit(10**6) 

class SegmentTree:
    def __init__(self, arr, func = lambda x, y : x + y, defaultvalue = 0) :
        self.n = len(arr)
        self.segmentTree = [0]*self.n + arr
        self.func = func
        self.defaultvalue = defaultvalue
        self.buildSegmentTree(arr)

    def buildSegmentTree(self, arr) :   
        for i in range(self.n -1, 0, -1) :
            self.segmentTree[i] = self.func(self.segmentTree[2*i] , self.segmentTree[2*i+1])         
            
    def query(self, l, r) :
        l += self.n
        r += self.n
        res = self.defaultvalue
        while l < r :
            if l & 1 :   
                res = self.func(res, self.segmentTree[l])
                l += 1
            l >>= 1
            if r & 1 :  
                r -= 1      
                res = self.func(res, self.segmentTree[r]) 
            r >>= 1
        return res

    def update(self, i, value) :
        i += self.n
        self.segmentTree[i] = value  
        while i > 1 :
            i >>= 1         
            self.segmentTree[i] = self.func(self.segmentTree[2*i] , self.segmentTree[2*i+1])


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