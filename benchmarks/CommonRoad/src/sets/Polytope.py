import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import pypoman
import scipy
from scipy.spatial import ConvexHull
from src.sets.Polygon import Polygon

class Polytope:
    """class representing a polytope in halfspace representation P = {x | c * x <= d}"""

    def __init__(self, c, d, id=None):
        """class constructor"""

        self.c = c                          # matrix C for the inequality constraint C*x <= d
        self.d = d                          # constant offset for the inequality constraint C*x <= d
        self.id = id

    def intersects(self, p):
        """check if the polytope intersects a point cloud p"""
        tmp = np.dot(self.c, p) - np.dot(np.resize(self.d, (self.c.shape[0], 1)), np.ones((1, p.shape[1])))
        return not np.all(np.max(tmp, 0) > 0)

    def intersection(self, P):
        """compute the intersection between two polytopes"""

        d1 = np.resize(self.d, (self.c.shape[0], 1))
        d2 = np.resize(P.d, (P.c.shape[0], 1))
        c = np.concatenate((self.c, P.c), axis=0)
        d = np.concatenate((d1, d2), axis=0)
        return Polytope(c, d)

    def shift(self, p):
        """shift the polytope by a vector p"""

        self.d = self.d + np.dot(self.c, p)

    def rotate(self, phi):
        """rotate the polytope by the angle phi"""

        self.c = np.dot(self.c, np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]))

    def vertices(self):
        """compute the vertices of a polytope"""

        if self.c.shape[1] == 2:
            return self.vertices2D()
        else:
            v = pypoman.compute_polytope_vertices(self.c, self.d)
            v_ = np.resize(v[0], (len(v[0]), 1))
            for i in range(1, len(v)):
                v_ = np.concatenate((v_, np.resize(v[i], (len(v[i]), 1))), axis=1)
            return v_

    def vertices2D(self):
        """compute vertices of a 2D polytope"""

        # normalize the polytope
        p = self.center()
        d = self.d + np.dot(self.c, -p)
        c = np.dot(np.diag(1/d.flatten()), self.c)

        # compute convex hull
        tmp = scipy.spatial.ConvexHull(c)

        # construct polytope vertices
        v = []
        v_ = np.concatenate((tmp.vertices, np.array([tmp.vertices[0]])))

        for i in range(len(v_)-1):
            diff = c[v_[i+1], :] - c[v_[i], :]
            tmp = np.array([[diff[1], -diff[0]]])
            d = np.dot(tmp, c[v_[i], :])
            if d < 0:
                v.append((-tmp / d).flatten())
            else:
                v.append((tmp / d).flatten())

        return np.asarray(v).T + p

    def center(self):
        """compute Chebychev center of a polytope"""

        tmp = np.expand_dims(np.linalg.norm(self.c, axis=1), axis=1)
        c = np.concatenate((self.c, tmp), axis=1)
        cost = np.zeros(self.c.shape[1] + 1)
        cost[-1] = -1.

        x = cp.Variable((len(cost), 1))
        prob = cp.Problem(cp.Minimize(np.expand_dims(cost, axis=0) @ x), [c @ x <= self.d])
        prob.solve(solver="ECOS")
        # solve with another solver for optimal_inaccurate
        if prob.status != "optimal":
            prob.solve(solver="MOSEK")
            

        assert prob.status == "optimal", f"id={self.id}, status={prob.status}"
        p = x.value[:-1]

        if not np.all(np.dot(self.c, p) <= self.d):
            raise Exception('Computation of the Chebychev center failed!')

        return p

    def polygon(self):
        """convert a 2D polytope to a polygon"""

        V = self.vertices()

        return Polygon(V[0, :], V[1, :])

    def plot(self, color):
        """plot the polytope"""

        v = pypoman.compute_polytope_vertices(self.c, self.d)
        v_ = np.resize(v[0], (len(v[0]), 1))
        for i in range(1, len(v)):
            v_ = np.concatenate((v_, np.resize(v[i], (len(v[i]), 1))), axis=1)
        v = np.concatenate((v_, v_[:, [0]]), axis=1)
        hull = ConvexHull(np.transpose(v))
        for simplex in hull.simplices:
            plt.plot(v[0, simplex], v[1, simplex], color)

    def normalize(self):
        """normalize the length of the halfspace normal vectors to one"""

        l = np.sqrt(np.sum(self.c ** 2, axis=1))
        A = np.diag(1/l)
        c = A @ self.c
        d = A @ self.d

        return Polytope(c, d)