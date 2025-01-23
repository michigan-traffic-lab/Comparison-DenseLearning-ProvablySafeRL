import numpy as np

class Interval:
    """class representing a interval"""

    def __init__(self, lb, ub):
        """class constructor"""

        self.lb = lb
        self.ub = ub

    def __mul__(self, factor):
        """multiplication of an interval with a scalar"""

        self.lb = self.lb * factor
        self.ub = self.ub * factor

    def __add__(self, p):
        """shift the interval by a constant offset"""

        self.lb = self.lb + p
        self.ub = self.ub + p

    def center(self):
        """compute center of the interval"""

        return 0.5 * (self.lb + self.ub)

    def width(self):
        """compute width of the interval"""

        return 0.5 * (self.ub - self.lb)

    def contains(self, point):
        """check if the interval contains a point"""

        return np.all(point >= self.lb) and np.all(point <= self.ub)

    def intersects(self, int):
        """check if two intervals intersect"""

        if self.lb <= int.lb:
            if int.lb >= self.ub:
                return False
        else:
            if self.lb >= int.ub:
                return False

        return True