import matplotlib.pyplot as plt
import numpy as np
from src.sets.Interval import Interval

class Zonotope:
    """class representing a zonotope Z = {c + sum_i G[:, i] * alpha_i | alpha_i in [-1,1]}"""

    def __init__(self, c, G):
        """class constructor"""

        self.c = c
        self.G = G

    def interval(self):
        """enclose the zonotope by an interval"""

        tmp = np.expand_dims(np.sum(abs(self.G), axis=1), axis=1)

        return Interval(self.c - tmp, self.c + tmp)

    def plot(self, color):
        """plot a zonotope"""

        # remove zero generators
        tmp = np.sum(abs(self.G), axis=0)
        ind = np.where(tmp > 0)[0]
        G = self.G[:, ind]
        c = self.c

        # size of enclosing interval
        xmax = np.sum(abs(G[0, :]))
        ymax = np.sum(abs(G[1, :]))

        # flip directions of generators so that all generators are pointing up
        ind = np.where(G[1, :] < 0)
        G[:, ind] = - G[:, ind]

        # sort generators according to their angles
        ang = np.arctan2(G[1, :], G[0, :])
        ind = np.where(ang < 0)[0]
        ang[ind] = ang[ind] + 2 * np.pi

        ind = np.argsort(ang)

        # sum the generators in the order of their angle
        n = G.shape[1]
        points = np.zeros((2, n+1))

        for i in range(n):
            points[:, i+1] = points[:, i] + 2 * G[:, ind[i]]

        points[0, :] = points[0, :] + xmax - np.max(points[0, :])
        points[1, :] = points[1, :] - ymax

        # mirror upper half of the zonotope to get the lower half
        tmp1 = np.concatenate((points[0, :], points[0, n] + points[0, 0] - points[0, 1:n+1]))
        tmp2 = np.concatenate((points[1, :], points[1, n] + points[1, 0] - points[1, 1:n+1]))

        tmp1 = np.resize(tmp1, (1, len(tmp1)))
        tmp2 = np.resize(tmp2, (1, len(tmp2)))

        points = np.concatenate((tmp1, tmp2), axis=0)

        # shift vertices by the center vector
        points = points + c

        plt.plot(points[0, :], points[1, :], color)
