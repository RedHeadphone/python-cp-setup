import sys, random, math, string, itertools
from copy import deepcopy
from collections import defaultdict, Counter, deque, OrderedDict
from heapq import heapify, heappush, heappop
from functools import cache, reduce
from bisect import bisect_left, bisect_right
from types import GeneratorType
from typing import *

input = lambda : sys.stdin.readline().strip()

def debug(*x,**y):
    if not DEBUG_ENABLED: return
    print(*x,(y if y!={} else "\b"), file=sys.stderr)


class SegmentTree:
    def __init__(self, arr, func = lambda x, y : x + y, defaultvalue = 0) :
        self.n = len(arr)
        self.segmentTree = [0]*self.n + arr
        self.func = func
        self.defaultvalue = defaultvalue
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
    def __init__(self, type = dict, arg = []): # allowed types: dict, defaultdict, Counter
        self.RANDOM = random.randrange(2**62)
        if type != defaultdict: self.dict = type([self.wrapper(i) for i in arg])
        else: self.dict = type(arg) # arg should be function
    def wrapper(self,num):
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
        return [(self.wrapper(i),j) for i,j in self.dict.items()]
    def __repr__(self):
        return repr({self.wrapper(i):j for i,j in self.dict.items()})

class CustomSet:
    def __init__(self, arr = []):
        self.RANDOM = random.randrange(2**62)
        self.set = set([self.wrapper(i) for i in arr])
    def wrapper(self,num):
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
    def __init__(self, base = 256, string = "", func = ord):
        self.base = base
        self.hash = 0
        self.length = 0
        self.func = func
        for i in string:
            self.right_add(i)

    def right_add(self, char):
        self.hash =  (self.hash * self.base + self.func(char)) % MOD
        self.length += 1
    
    def left_add(self, char):
        self.hash =  (self.hash + self.func(char) * pow(self.base, self.length, MOD)) % MOD
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


def binary_search(left,right,check,start_from_left):
    if start_from_left: ans = left 
    else: ans = right
    while left<=right:
        mid = (left+right)//2
        if start_from_left:
            if check(mid):
                ans,left = mid, mid+1
            else:
                right = mid-1
        else:
            if check(mid):
                ans,right = mid, mid-1
            else:
                left = mid+1
    return ans


class BitManipulation:
    def bin(self,num,size=None):
        bin_string = bin(num)[2:]
        return bin_string if (size==None) else bin_string.zfill(size)

    def bit_sum(self,num): # returns array of sum of bits of all numbers from 0 to num
        b = len(self.bin(num))
        bs = [0] * b
        c = 2**b - 1
        for i in range(b):
            c -= (1<<i)
            bs[i] = (num & c) // 2
            if num & ((1<<i)):
                bs[i] += (num % (1<<i)) + 1
        return bs
    

class Maths:
    def __init__(self, fact_memo=False, lpf_memo=False, max_n=(2*(10**5)+5)):
        if fact_memo:
            fact = [1]*max_n
            invfact = [1]*max_n
            for i in range(1,max_n):
                fact[i] = (fact[i-1]*i)%MOD
                invfact[i] = (invfact[i-1]*self.mod_inverse(i))%MOD
            self.fact = fact
            self.invfact = invfact
        
        if lpf_memo:
            last_prime_factor = [0]*max_n
            for i in range(2, max_n):
                if last_prime_factor[i] > 0: continue
                for j in range(i, max_n, i):
                    last_prime_factor[j] = i
            self.last_prime_factor = last_prime_factor

    def ncr(self,n, r):
        if n < r: return 0
        return (self.fact[n]*self.invfact[r]*self.invfact[n-r])%MOD

    def npr(self,n, r):
        if n < r: return 0
        return (self.fact[n]*self.invfact[n-r])%MOD

    def mod_inverse(self,a):
        return pow(a,MOD-2,MOD)

    def ncr_without_memo(self,n, r):
        if n < r: return 0
        num = den = 1
        for i in range(r):
            num = (num * (n - i)) % MOD
            den = (den * (i + 1)) % MOD
        return (num * pow(den,
                MOD - 2, MOD)) % MOD

    def npr_without_memo(self,n, r):
        if n < r: return 0
        num = 1
        for i in range(n,n-r,-1):
            num = (num * i) % MOD
        return num

    def factors(self,n): 
        if n==0:
            return set()   
        return set(reduce(list.__add__, 
                    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

    def prime_factors_using_lpf(self,n):
        factors = set()
        while n > 1:
            factors.add(self.last_prime_factor[n])
            n //= self.last_prime_factor[n]
        return factors

    def prime_factors_with_power(self,n):
        factors = []
        divisor = 2
        while divisor * divisor <= n:
            power = 0
            while n % divisor == 0:
                n //= divisor
                power += 1
            if power > 0: factors.append((divisor, power))
            divisor += 1
        if n > 1: factors.append((n, 1))
        return factors

    def is_prime(self,n):
        if n==1:
            return False
        for i in range(2,int(n**0.5)+1):
            if n%i==0:
                return False
        return True


###############################################################################

def solve(case=None):
    pass

###############################################################################

MULTIPLE_CASES = 1
DEBUG_ENABLED = 0
BOOLEAN_RETURN = 0
MOD = 10**9 + 7
TRUE_MAPPING,FALSE_MAPPING = 'YES','NO'

number_of_test_cases = 1 if not MULTIPLE_CASES else int(input())
for t in range(number_of_test_cases):
    if BOOLEAN_RETURN: print(TRUE_MAPPING if solve(t+1) else FALSE_MAPPING)
    else: solve(t+1)