import numpy as np
from src.sets.Zonotope import Zonotope

class PolyZonotope:
    """class representing a polynomial zonotope
    PZ = {c + sum_i prod_k G[:,i] alpha_i^E[i,k] + sum_j G_rest[:,j] beta_j | alpha_i, beta_j in [-1,1]}"""

    def __init__(self, c, G, E, Grest):
        """class constructor"""

        self.c = c
        self.G = G
        self.E = E
        self.Grest = Grest

    def zonotope(self):
        """enclose polynomial zonotope by a zonotope"""

        tmp = np.prod(np.ones(self.E.shape) - np.mod(self.E, 2), axis=0)
        ind = [i for i in range(0, len(tmp)) if tmp[i] == 1]
        ind_ = np.setdiff1d(np.arange(0, len(tmp)), ind)

        c = self.c + 0.5*np.resize(np.sum(self.G[:, ind], axis=1), self.c.shape)
        G = np.concatenate((self.G[:, ind_], 0.5*self.G[:, ind], self.Grest), axis=1)

        return Zonotope(c, G)

    def replace_factor(self, ind, val):
        """replace a factor of the polynomial zonotope with a fixed value"""

        G = self.G * (val**self.E[ind, :])
        E = np.delete(self.E, ind, 0)

        return PolyZonotope(self.c, G, E, self.Grest).compact()

    def compact(self):
        """remove redundancies from a polynomial zonotope"""

        # sort the columns of the exponent matrix
        ind = np.lexsort(np.transpose(self.E).T)
        E = self.E[:, ind]
        G = self.G[:, ind]
        c = self.c

        # remove all-zero columns in the exponent matrix
        if np.sum(E[:, 0]) == 0:
            c = c + G[:, [0]]
            E = E[:, 1:]
            G = G[:, 1:]

        # remove all duplicate columns in the exponent matrix
        cnt = 0
        E_new = np.zeros(E.shape).astype(int)
        G_new = np.zeros(G.shape)

        if E.shape[1] > 0:

            E_new[:, cnt] = E[:, 0]
            G_new[:, cnt] = G[:, 0]

            for i in range(1, E.shape[1]):
                if np.all(E_new[:, cnt] == E[:, i]):
                    G_new[:, cnt] = G_new[:, cnt] + G[:, i]
                else:
                    cnt += 1
                    G_new[:, cnt] = G[:, i]
                    E_new[:, cnt] = E[:, i]

        return PolyZonotope(c, G_new[:, 0:cnt+1], E_new[:, 0:cnt+1], self.Grest)
