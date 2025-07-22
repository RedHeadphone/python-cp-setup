import sys, random, math, string, itertools
from copy import deepcopy
from collections import defaultdict, Counter, deque
from heapq import heapify, heappush, heappop, nlargest, nsmallest
from functools import cache
from bisect import bisect_left, bisect_right
from types import GeneratorType
from typing import *

input = lambda: sys.stdin.readline().strip()


def debug(*x, **y):
    if not DEBUG_ENABLED:
        return
    print(*(list(x)+([y] if y != {} else [])), file=sys.stderr)


###############################################################################


def solve_bruteforce(case=None):
    pass


def execute_once():
    pass


def solve(case=None):
    pass


INPUT_NUMBER_OF_TEST_CASES = 1
DEBUG_ENABLED = 0
BOOLEAN_RETURN = 0
TRUE_MAPPING, FALSE_MAPPING = "Yes", "No"
MOD = 10**9 + 7

###############################################################################


class CustomHashMap:
    def __init__(self, map_type=dict, arg=[]):
        """
        Custom HashMap for Python's Anti-hash-table test
        :param map_type: type of the map (allowed types: dict, defaultdict, Counter)
        :param arg: argument for the map (should be list if map_type is Counter and function if map_type is defaultdict)
        """
        self.RANDOM = random.randrange(2**62)
        if map_type == Counter:
            self.dict = map_type([self.wrapper(i) for i in arg])
        elif map_type == defaultdict:
            self.dict = map_type(arg)
        else:
            self.dict = map_type()

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


def recursion_limit_fix(f):
    """
    recursion limit fix decorator, change 'return' to 'yield' and add 'yield' before calling the function
    and add 'yield' after the function if not returning anything
    """
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


def sorted_unique(arr):
    arr = arr[::]
    arr.sort()
    res = []
    for i in arr:
        if not res or res[-1] != i:
            res.append(i)
    return res


execute_once()

if __name__ == "__main__":
    for t in range(int(input()) if INPUT_NUMBER_OF_TEST_CASES else 1):
        if BOOLEAN_RETURN:
            print(TRUE_MAPPING if solve(t + 1) else FALSE_MAPPING)
        else:
            solve(t + 1)
