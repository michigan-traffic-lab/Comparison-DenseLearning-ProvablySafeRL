from shapely import geometry
import numpy as np
import matplotlib.pyplot as plt

class Polygon:
    """class representing a polygon"""

    def __init__(self, x, y):
        """class constructor"""

        self.set = geometry.Polygon([*zip(x, y)])
        self.V = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)

    def intersects(self, pgon):
        """check if two polygons intersect"""

        return self.set.intersects(pgon.set)

    def plot(self, color):
        """plot the polygon"""

        v = np.concatenate((self.V, self.V[:, [0]]), axis=1)
        plt.plot(v[0, :], v[1, :], color)

    def convex_hull(self):
        """compute the convex hull of the polygon"""

        self.set = self.set.convex_hull
        x, y = self.set.exterior.coords.xy
        self.V = np.concatenate((np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)), axis=0)

        return self