import random
import math
from collections import defaultdict, Counter, deque, OrderedDict
from heapq import heapify, heappush, heappop
from functools import lru_cache, reduce
from bisect import bisect_left, bisect_right
from types import GeneratorType
import sys

input = lambda : sys.stdin.readline().strip()
# sys.setrecursionlimit(10**6)    # doesn't work on codeforces

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
        self.count = [1]*n
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
            self.count[y] += self.count[x]

dire = [0,1,0,-1,0]

def is_prime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i==0:
            return False
    return True

def ncr(n, r, p):
    num = den = 1
    for i in range(r):
        num = (num * (n - i)) % p
        den = (den * (i + 1)) % p
    return (num * pow(den,
            p - 2, p)) % p

def case(t):
    print("Case #{}:".format(t), end=" ")

# For codeforces - hashing function 
RANDOM = random.randrange(2**62)
def Wrapper(x):
  return x ^ RANDOM

def factors(n): 
    if n==0:
        return set()   
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

# MAX = 10**5+5
# factors = [factors(i) for i in range(MAX)]

MOD = 10**9 + 7

def binpow(a, b):
    if b==0:
        return 1
    res = binpow(a,b//2)
    res = pow(res,2,MOD)
    if b%2:
        return (res*a)%MOD
    return res

def mod_inverse(a):
    return binpow(a,MOD-2)

MAX = 2*(10**5)+5

# fact = [1]*MAX
# invfact = [1]*MAX
# for i in range(1,MAX):
#     fact[i] = (fact[i-1]*i)%MOD
#     invfact[i] = (invfact[i-1]*mod_inverse(i))%MOD

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

###############################################################################


def solve():
    n = int(input())
    mapp = defaultdict(list)
    for i in range(n-1):
        a,b = map(int,input().split())
        mapp[a].append(b)
        mapp[b].append(a)
    q = int(input())
    queries = []
    for i in range(q):
        a,b = map(int,input().split())
        queries.append((a,b))
    ans_mapp = {}

    @bootstrap
    def dfs(node, parent = None):
        leaf_nodes = 0
        child_count = 0
        for child in mapp[node]:
            if child!=parent:
                child_count += 1
                leaf_nodes += (yield dfs(child,node))
        if child_count == 0:
            leaf_nodes = 1
        ans_mapp[Wrapper(node)] = leaf_nodes
        yield leaf_nodes

    dfs(1)

    for a,b in queries:
        print(ans_mapp[Wrapper(a)]*ans_mapp[Wrapper(b)])



###############################################################################

for t in range(int(input())):
    # case(t+1)
    solve()