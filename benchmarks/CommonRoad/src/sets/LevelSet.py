import numpy as np
from src.sets.Zonotope import Zonotope

class LevelSet:
    """class representing a polynomial level set LS = { x | sum_i prod_k a[i] x[i]^E[i,k] <= b}"""

    def __init__(self, a, b, E):
        """class constructor"""

        self.a = a
        self.b = b
        self.E = E

    def contains(self, p):
        """check if the level set contains a point p"""

        tmp = np.expand_dims(np.sum(self.a * np.prod(p ** self.E, axis=0), axis=1), axis=1) - self.b

        return np.min(tmp) <= 0

    def normalize(self):
        """normalize the length of the coefficient vectors to one"""

        l = np.sqrt(np.sum(self.a ** 2, axis=1))
        D = np.diag(1/l)
        a = D @ self.a
        b = D @ self.b

        return LevelSet(a, b, self.E)
