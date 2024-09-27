from main import MOD, Counter


class Combinatorics:
    def __init__(self, pre_compute_limit=0):
        self.fact = [1] * pre_compute_limit
        self.invfact = [1] * pre_compute_limit
        for i in range(1, pre_compute_limit):
            self.fact[i] = (self.fact[i - 1] * i) % MOD
            self.invfact[i] = (self.invfact[i - 1] * pow(i, MOD - 2, MOD)) % MOD

    def ncr(self, n, r):
        if n < r or r < 0:
            return 0
        return (self.fact[n] * self.invfact[r] * self.invfact[n - r]) % MOD

    def npr(self, n, r):
        if n < r or r < 0:
            return 0
        return (self.fact[n] * self.invfact[n - r]) % MOD

    def ncr_without_memo(self, n, r):
        if n < r or r < 0:
            return 0
        num = den = 1
        for i in range(r):
            num = (num * (n - i)) % MOD
            den = (den * (i + 1)) % MOD
        return (num * pow(den, MOD - 2, MOD)) % MOD

    def npr_without_memo(self, n, r):
        if n < r or r < 0:
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
