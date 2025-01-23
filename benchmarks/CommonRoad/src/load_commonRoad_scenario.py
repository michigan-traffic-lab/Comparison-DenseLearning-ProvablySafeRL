import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Rectangle, Polygon
from commonroad.scenario.scenario import Scenario
from pypoman import compute_polytope_halfspaces

import os

from src.sets.Interval import Interval
from src.sets.Polytope import Polytope
from src.auxiliary.Obstacle import Obstacle


def occupancy_to_obstacle(occupancy, dt, id=None):
    time = Interval((occupancy.time_step.start) * dt, occupancy.time_step.end * dt)
    set = obstacle2polytope(occupancy.shape, id=id)

    return Obstacle(set, time)


def find_highD_road_boundary(lanelet_network):
    y_min, y_max = None, None
    for lanelet in lanelet_network.lanelets:
        if lanelet.adj_left is None:
            y_max = lanelet.left_vertices[0][1]
            break
    for lanelet in lanelet_network.lanelets:
        if lanelet.adj_right is None:
            y_min = lanelet.right_vertices[0][1]
            break
    return y_min, y_max


def load_commonRoad_scenario(scenario: Scenario, only_load_static=True):
    """load a CommonRoad scenario and bring it to the correct format"""

    obs = []

    # load commonRoad scenario
    dt = scenario.dt

    if not only_load_static:
        # convert dynamic obstacles to polytopes
        for d in scenario.dynamic_obstacles:
            for o in d.prediction.occupancy_set:
                obs.append(occupancy_to_obstacle(o, dt))

    # convert static obstacles to polytopes
    for o in scenario.static_obstacles:
        set = obstacle2polytope(o.obstacle_shape, id="static")
        set.rotate(o.initial_state.orientation)
        set.shift(np.reshape(o.initial_state.position, (2, 1)))
        obs.append(Obstacle(set))

    y_min, y_max = find_highD_road_boundary(scenario.lanelet_network)
    # convert road-boundary to polytopes (hardcoded for now)
    obs.append(Obstacle(Polytope(np.array([[0.0, 1.0]]), np.array([[y_min+0.01]]), id="road"))) # add margin value to avoid rounding errors
    obs.append(Obstacle(Polytope(np.array([[0.0, -1.0]]), np.array([[-y_max+0.01]]), id="road")))


    return obs

def obstacle2polytope(shape: Union[Rectangle, Polygon], id=None):
    """convert a CommonRoad shape object to a polytope"""

    if isinstance(shape, Rectangle):

        C = np.concatenate((np.identity(2), -np.identity(2)), axis=0)
        d = np.array([[shape.length/2], [shape.width/2], [shape.length/2], [shape.width/2]])

        poly = Polytope(C, d)
        poly.rotate(shape.orientation)
        poly.shift(np.reshape(shape.center, (2, 1)))

    elif isinstance(shape, Polygon):

        V = shape.vertices.T

        # remove redundant vertices
        ind = []

        for i in range(1, V.shape[1]):
            if np.linalg.norm(V[:, i - 1] - V[:, i]) < 10 ** (-6):
                ind.append(i)

        V = np.delete(V, ind, axis=1)

        # convert polygon to polytope
        C = np.zeros((V.shape[1]-1, 2))
        d = np.zeros((V.shape[1]-1, 1))

        c = np.mean(V, axis=1)

        for i in range(V.shape[1]-1):
            dir = V[:, i] - V[:, i+1]
            C[i, 0] = -dir[1]
            C[i, 1] = dir[0]
            d[i] = np.dot(C[[i], :], V[:, [i]])

            if np.dot(C[[i], :], c) > d[i]:
                C[i, :] = -C[i]
                d[i, :] = -d[i]

        poly = Polytope(C, d, id=id)

    return poly
