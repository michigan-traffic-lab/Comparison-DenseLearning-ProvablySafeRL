import torch
import pickle
import os, sys
sys.path.append((os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
import numpy as np
from copy import deepcopy
from src.auxiliary.utils import MultiLayerPerceptron
from src.auxiliary.ReachableSetManager import ReachableSetManager

class UnsafeReinforcementPlanner:
    """class representing the unsafe reinforcement learning motion planner"""

    def __init__(self, benchmark, model: MultiLayerPerceptron):
        """class constructor"""
        self.model = model
        self.reachable_set_manager = ReachableSetManager(benchmark)

    def plan(self, observation_dict):
        """determine control commands from the current observation using the learned policy"""
        normalized_observation_vector = observation_dict["normalized_observation_vector"]
        alpha = self.model.forward(normalized_observation_vector)
        alpha = np.expand_dims(alpha, axis=1)

        return alpha
