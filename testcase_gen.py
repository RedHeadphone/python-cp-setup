import random

def random_string(n = None, p = True):
    if n==None:
        n = random.randint(5, 10)
    if p:
        print(n)
    for _ in range(n):
        print(chr(random.randint(97, 122)), end="")

def random_array(n = None, p = True):
    if n==None:
        n = random.randint(5, 10)
    if p:
        print(n)
    for _ in range(n):
        print(random.randint(1, 10), end=" ")

def random_binary_string(n = None, p = True):
    if n==None:
        n = random.randint(5, 10)
    if p:
        print(n)
    for _ in range(n):
        print(random.choice("01"), end="")

def random_matrix(n=None, m=None, p = True):
    if n==None:
        n = random.randint(5, 10)
    if m==None:
        m = random.randint(5, 10)
    if p:
        print(n, m)
    for _ in range(n):
        for _ in range(m):
            print(random.randint(1, 10), end=" ")
        print()

t = random.randint(5, 20)
print(t)


for _ in range(t):
    
    print()