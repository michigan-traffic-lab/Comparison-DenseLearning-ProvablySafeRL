class Obstacle:
    """class representing a static or dynamic obstacle"""

    def __init__(self, set, time=None):
        """class constructor"""

        self.set = set                              # set representing the shape of the obstacle
        self.time = time                            # time interval in which the obstacle is active