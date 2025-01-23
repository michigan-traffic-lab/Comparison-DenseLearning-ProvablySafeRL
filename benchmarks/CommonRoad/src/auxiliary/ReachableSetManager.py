import os
import scipy.io
import scipy
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from src.sets.PolyZonotope import PolyZonotope
from src.sets.Polytope import Polytope
from src.sets.Interval import Interval
from src.sets.Polygon import Polygon
from src.sets.Zonotope import Zonotope


class ReachSet:
    """class representing a reachable set"""

    def __init__(self, occ, pgon, time, U, Xinit, ind):
        """class constructor"""

        self.occ = occ                      # reachable set (occupancy set) represented as polynomial zonotope
        self.pgon = pgon                    # reachable set (occupancy set) represented as a polygon
        self.time = time                    # time intervals for the reachable set
        self.U = U                          # control input zonotope
        self.Xinit = Xinit                  # initial set for the non-invariant states
        self.ind = ind                      # indices of the zonotope factors that belong to control inputs


class ReachableSetManager:
    """class to load reachable set and project alpha to control inputs"""

    def __init__(self, benchmark):

        # load pre-computed reachable sets
        path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(path, 'reachability', 'data', benchmark)
        self.R, self.N, self.names = self.import_reachable_sets(path)
        self.U = self.R[0].U

    @staticmethod
    def import_reachable_sets(path):
        """import the pre-computed reachable set templates"""

        cnt_file = 1
        R = []
        dirname = os.path.dirname(os.path.dirname(__file__))

        while True:

            cnt_set = 1
            pgon = []
            occ = []
            time = []

            while True:

                filename = 'reach_' + str(cnt_file) + '_' + str(cnt_set)
                filepath = os.path.join(dirname, path, filename)
                try:
                    mat = scipy.io.loadmat(filepath)
                except:
                    break

                occ.append(PolyZonotope(mat['c'], mat['G'], mat['E'], mat['Grest']))
                pgon.append(Polygon(mat['V'][0, :], mat['V'][1, :]))
                N = mat['N'][0][0]
                dt = mat['dt'][0][0]
                time.append(Interval(cnt_set-1, cnt_set))
                cnt_set = cnt_set + 1

            if len(occ) == 0:
                break

            for i in range(len(time)):
                time[i] * ((N*dt)/len(time))

            U = Zonotope(mat['c_u'], mat['G_u'])
            Xinit = Interval(mat['l'], mat['u'])

            names = mat['names'][0, :].tolist()
            for i in range(len(names)):
                names[i] = names[i][0]

            ind = np.arange(0, N*mat['G_u'].shape[1])

            R.append(ReachSet(occ, pgon, time, U, Xinit, ind))
            cnt_file = cnt_file + 1

        return R, N, names

    def select_reachable_set(self, observation):
        """select the correct reachable set template based on the current observation"""

        # map observations to non-invariant states
        x0 = np.zeros((len(self.names), 1))

        for i in range(len(self.names)):
            x0[i] = observation[self.names[i]]

        # select reachable set
        for i in range(len(self.R)):
            if self.R[i].Xinit.contains(x0):
                R = self.R[i]
                break

        # reduce ranges of the non-invariant states to the actual values
        val = (x0 - R.Xinit.center())/R.Xinit.width()
        val[np.isnan(val)] = 0
        val = np.reshape(val, [len(val), ])

        occ = []

        for o in deepcopy(R.occ):
            for i in range(len(val)):
                o = o.replace_factor(0, val[i])
            occ.append(o)

        return ReachSet(occ, R.pgon, R.time, R.U, R.Xinit, R.ind)

    def factor2control(self, alpha):
        """compute the control input u from the zonotope factors alpha"""
        u = []
        n_u = self.U.G.shape[0]
        for i in range(self.N):
            alpha_ = alpha[i * n_u:(i + 1) * n_u]
            u_ = self.U.c + np.dot(self.U.G, np.reshape(alpha_, (len(alpha_), 1)))
            u.append(np.squeeze(u_))
        return u

    def control2factor(self, u):
        """compute the zonotope factors alpha form control input u"""
        alpha = []
        n_u = self.U.G.shape[0]

        for i in range(self.N):
            u_ = u[i * n_u:(i + 1) * n_u]
            alpha_ = scipy.linalg.solve(self.U.G, (u_-self.U.c.reshape(2)))
            alpha.append(np.squeeze(alpha_))

        return alpha