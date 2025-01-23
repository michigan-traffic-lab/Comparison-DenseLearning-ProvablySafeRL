import copy
import logging
import math
import random
import sys
import time
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Union
from xml.etree import cElementTree as ET
import numpy as np

from commonroad.geometry.shape import Rectangle
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.util import Interval
from commonroad.prediction.prediction import TrajectoryPrediction
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.obstacle import DynamicObstacle
from commonroad.scenario.scenario import Scenario
from commonroad.scenario.trajectory import State, Trajectory

import sumocr
from sumocr.interface.ego_vehicle import EgoVehicle
from sumocr.interface.util import *
from sumocr.maps.scenario_wrapper import AbstractScenarioWrapper
from sumocr.maps.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config import DefaultConfig

import libsumo as traci
from gym_commonroad_sumo.simulation.sumo_simulation import SumoSimulation

from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.simulator import Simulator
from gym_commonroad_sumo.NDE_RL_NADE.envs.nade import NADE
from gym_commonroad_sumo.NDE_RL_NADE.controller.treesearchnadecontroller import (
    TreeSearchNADEBackgroundController,
)
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.network.trafficnet import TrafficNet


class NADESimulation(SumoSimulation):
    def initialize(
        self,
        conf: DefaultConfig,
        scenario_wrapper: ScenarioWrapper,
        planning_problem_set: PlanningProblemSet = None,
        sumo_initialization=None,
        output_file=None,
        episode=1,
        sumo_net_file=None,
        worker_id=0,
        single_input_path=None,
    ) -> None:
        """
        Reads scenario files, starts traci simulation, initializes vehicles, conducts pre-simulation.

        :param conf: configuration object. If None, use default configuration.
        :param scenario_wrapper: handles all files required for simulation. If None it is initialized with files
            folder conf.scenarios_path + conf.scenario_name
        :param planning_problem_set: initialize initial state of ego vehicles
        (if None, use planning_problem_set from self.scenario)

        """
        if conf is not None:
            self.conf = conf

        self.logger = self._init_logging()

        assert isinstance(
            scenario_wrapper, AbstractScenarioWrapper
        ), f"scenario_wrapper expected type ScenarioWrapper or None, but got type {type(scenario_wrapper)}"
        self.scenarios = scenario_wrapper
        self.dt = self.conf.dt
        self.dt_sumo = self.conf.dt / self.conf.delta_steps
        self.delta_steps = self.conf.delta_steps
        self.planning_problem_set = (
            planning_problem_set
            if planning_problem_set is not None
            else self.scenarios.planning_problem_set
        )

        assert (
            sumocr.sumo_installed
        ), "SUMO not installed or environment variable SUMO_HOME not set."
        self.input = None
        self.env = NADE(
            BVController=TreeSearchNADEBackgroundController,
            cav_model="CR",
        )

        # initialize sumo simulator
        if single_input_path is None:
            self.sim = Simulator(
                sumo_net_file_path = sumo_net_file,
                sumo_config_file_path = self.scenarios.sumo_cfg_file,
                num_tries=50,
                step_size=0.1,
                action_step_size=0.1,
                lc_duration=1,
                track_cav=False,
                sublane_flag=True,
                gui_flag=self.conf.with_sumo_gui,
                output=["fcd"],
                experiment_path=output_file,
                worker_id=worker_id,
            )
        else:
            self.sim = Simulator(
                sumo_net_file_path = sumo_net_file,
                sumo_config_file_path = self.scenarios.sumo_cfg_file,
                num_tries=50,
                step_size=0.1,
                action_step_size=0.1,
                lc_duration=1,
                track_cav=False,
                sublane_flag=True,
                gui_flag=self.conf.with_sumo_gui,
                output=["fcd"],
                experiment_path=output_file,
                worker_id=worker_id,
                replay_single=True,
                replay_flag=True,
                single_input_path=single_input_path,
            )
        self.sim.bind_env(self.env)
        self.start_nade(episode=episode, output_folder=output_file)
        self.initialized = True
        
        # initialize ego vehicles
        self.init_ego_vehicles_from_planning_problem(self.planning_problem_set)

        # initialize obstacles and update states
        self.env.initialize()
        traci.simulationStep()
        self.env._check_vehicle_list()
        new_vehicle_ids, new_ped_ids = self._init_vehicle_ids()
        self._fetch_sumo_vehicles(0, new_vehicle_ids)
        self._fetch_sumo_pedestrians(0, new_ped_ids)

    # start nade simulation
    def start_nade(self, episode=0, output_folder='.'):
        """Start SUMO simulation or initialize environment."""
        self.sim.split_run_flag = True
        self.sim.episode = episode
        self.sim.start()

    def simulate_step(self) -> None:
        """
        Executes next simulation step (consisting of delta_steps sub-steps with dt_sumo=dt/delta_steps) in SUMO
        """

        # simulate sumo scenario for delta_steps time steps
        for i in range(self.delta_steps):
            # send ego vehicles to SUMO
            if not self.dummy_ego_simulation and len(self.ego_vehicles) > 0:
                self._send_ego_vehicles(self.ego_vehicles, i)

            # execute SUMO simulation step
            self.sim.pre_step(self.sim)
            self.env.step()
            self.sim.sumo_time_stamp = self.sim.get_time()
            traci.simulationStep()
            self.sim.current_time_steps += 1
            self.env._check_vehicle_list()

            for ego_veh in list(self.ego_vehicles.values()):
                ego_veh._current_time_step += 1

        # get updated obstacles from sumo
        self._current_time_step += 1
        new_vehicle_ids, new_ped_ids = self._init_vehicle_ids()
        self._fetch_sumo_vehicles(self.current_time_step, new_vehicle_ids)
        self._fetch_sumo_pedestrians(self.current_time_step, new_ped_ids)

    def stop(self) -> None:
        """
        Stop simulation and close all connections.
        """
        self.sim.stop()
        self.initialized = False
