from main import random


class RollingHash:
    def __init__(
        self,
        string="",
        max_length=10**5,
        base=256,
        func=ord,
        mod1=10**9 + 7,
        mod2=10**9 + 9,
    ):
        self.base, self.func = base, func
        self.mod1, self.mod2 = mod1, mod2
        self.hash1, self.hash2 = 0, 0
        self.length = 0
        self.b1, self.b2 = [1] * max_length, [1] * max_length
        self.r1, self.r2 = pow(base, mod1 - 2, mod1), pow(base, mod2 - 2, mod2)
        for i in range(1, max_length):
            self.b1[i], self.b2[i] = (self.b1[i - 1] * base) % mod1, (
                self.b2[i - 1] * base
            ) % mod2
        for i in string:
            self.right_add(i)

    def right_add(self, char):
        self.hash1 = (self.hash1 * self.base + self.func(char)) % self.mod1
        self.hash2 = (self.hash2 * self.base + self.func(char)) % self.mod2
        self.length += 1

    def right_remove(self, char):
        self.hash1 = ((self.hash1 - self.func(char)) * self.r1) % self.mod1
        self.hash2 = ((self.hash2 - self.func(char)) * self.r2) % self.mod2
        self.length -= 1

    def update(self, i, value, old_value):
        self.hash1 = (
            self.hash1
            - self.b1[self.length - i - 1] * (self.func(old_value) - self.func(value))
        ) % self.mod1
        self.hash2 = (
            self.hash2
            - self.b2[self.length - i - 1] * (self.func(old_value) - self.func(value))
        ) % self.mod2

    def left_add(self, char):
        self.hash1 = (self.hash1 + self.func(char) * self.b1[self.length]) % self.mod1
        self.hash2 = (self.hash2 + self.func(char) * self.b2[self.length]) % self.mod2
        self.length += 1

    def left_remove(self, char):
        self.hash1 = (
            self.hash1 - self.func(char) * self.b1[self.length - 1]
        ) % self.mod1
        self.hash2 = (
            self.hash2 - self.func(char) * self.b2[self.length - 1]
        ) % self.mod2
        self.length -= 1

    def get_hash(self):
        return (self.hash1, self.hash2)


class RangeHash:
    def __init__(self, s, mod=2147483647, base1=None, base2=None):
        self.mod, self.base1, self.base2 = (
            mod,
            base1 if base1 is not None else random.randrange(mod),
            base2 if base2 is not None else random.randrange(mod),
        )
        _len = len(s)
        f_hash, f_pow = [0] * (_len + 1), [1] * (_len + 1)
        s_hash, s_pow = f_hash[:], f_pow[:]
        for i in range(_len):
            f_hash[i + 1] = (base1 * f_hash[i] + s[i]) % mod
            s_hash[i + 1] = (base2 * s_hash[i] + s[i]) % mod
            f_pow[i + 1] = base1 * f_pow[i] % mod
            s_pow[i + 1] = base2 * s_pow[i] % mod
        self.f_hash, self.f_pow = f_hash, f_pow
        self.s_hash, self.s_pow = s_hash, s_pow

    def get_hash(self, start, stop):
        return (
            (self.f_hash[stop] - self.f_pow[stop - start] * self.f_hash[start])
            % self.mod,
            (self.s_hash[stop] - self.s_pow[stop - start] * self.s_hash[start])
            % self.mod,
        )
