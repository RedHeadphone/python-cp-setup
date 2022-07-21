import random
import math
from collections import defaultdict, Counter
from functools import lru_cache
import bisect
import heapq

# import sys 
# sys.setrecursionlimit(10**6) 

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

def google(t):
    print("Case #{}:".format(t), end=" ")

RANDOM = random.randrange(2**62)
def Wrapper(x):
  return x ^ RANDOM

###############################################################################

def solve():
    pass

###############################################################################

for t in range(int(input())):
    # google(t+1)
    solve()