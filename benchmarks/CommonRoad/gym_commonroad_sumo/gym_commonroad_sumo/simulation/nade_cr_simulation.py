"""
NADE simulation specific helper methods
"""

import copy
import os
from typing import Dict
from commonroad.planning.planning_problem import PlanningProblemSet
from gym_commonroad_sumo.simulation.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config.default import DefaultConfig
from collections import OrderedDict
from gym_commonroad_sumo.utils.navigator import Navigator
from commonroad.planning.planning_problem import PlanningProblem
from gym_commonroad_sumo.action.vehicle import Vehicle

from gym_commonroad_sumo.simulation.simulations import (
    sumo_cr_simulator,
    simulate_scenario,
)
from gym_commonroad_sumo.simulation.sumo_scenario import ScenarioWrapper
from gym_commonroad_sumo.simulation.nade_simulation import NADESimulation

# NADE simulation
class simulate_scenario_nade(simulate_scenario):
    def __init__(
        self,
        obs_configs: Dict,
        conf: DefaultConfig,
        scenario_wrapper: ScenarioWrapper,
        planning_problem_set: PlanningProblemSet = None,
        planning_problem: PlanningProblem = None,
        sumo_initialization=None,
        ego_vehicle: Vehicle = None,
        reset_config=None,
        output_file=None,
        sumo_net_file=None,
        episode=1,
        worker_id=0,
        single_input_path=None,
    ):

        self._planning_problem_set = planning_problem_set
        self._planning_problem = planning_problem
        self._scenario_id = scenario_wrapper.initial_scenario.scenario_id
        self._scenario = scenario_wrapper.initial_scenario
        self.current_scenario = self.copy_scenario(self._scenario)
        self.ego_vehicle = ego_vehicle

        # create sumo simulation
        self.sumo_sim = NADESimulation()

        # initialize simulation
        self.sumo_sim.initialize(
            conf,
            scenario_wrapper,
            planning_problem_set,
            sumo_initialization,
            output_file,
            sumo_net_file=sumo_net_file,
            episode=episode,
            worker_id=worker_id,
            single_input_path=single_input_path,
        )
        initial_obstacles = self.sumo_sim._get_cr_obstacles_at_time(0)
        obstacle_dict = dict()
        for o in initial_obstacles:
            obstacle_dict[o.obstacle_id] = o
        self.current_scenario._dynamic_obstacles = obstacle_dict
        self.current_scenario_full = self.copy_scenario(self.current_scenario)

        # initialize basic components
        self.navigator: Navigator = None
        self._continous_collision_check = obs_configs["action_configs"].get(
            "continuous_collision_checking", True
        )
        self._max_lane_merge_range = obs_configs.get("max_lane_merge_range")
        if planning_problem is not None:
            self._goal_region = self._planning_problem.goal
            self.episode_length = max(
                s.time_step.end for s in self._goal_region.state_list
            )
        else:
            self._goal_region = None
            self.episode_length = None

        # initialize basic info
        self.ego_lanelet_id = None
        self.ego_lanelet = None
        self.time_step = 0
        self._update_collision_checker()
        self._create_navigator()
        self._get_lanelet_polygons()
        self.local_ccosy = None
        self._local_merged_lanelet = None
        self.lane_circ_sensor_range_radius: float = obs_configs.get(
            "lane_circ_sensor_range_radius", 100
        )
        self.max_obs_dist = self.lane_circ_sensor_range_radius
        self._road_edge = {
            "left_road_edge_lanelet_id_dict": reset_config[
                "left_road_edge_lanelet_id_dict"
            ],
            "right_road_edge_lanelet_id_dict": reset_config[
                "right_road_edge_lanelet_id_dict"
            ],
            "left_road_edge_dict": reset_config["left_road_edge_dict"],
            "right_road_edge_dict": reset_config["right_road_edge_dict"],
            "boundary_collision_object": reset_config["boundary_collision_object"],
        }

        # initialize observation space
        self.observation_space_size: int = None
        self._flatten_observation = obs_configs.get("flatten_observation")
        self.observation_dict = OrderedDict()
        self.observation_history_dict: dict = dict()
        self.observation_space = self._build_observation_space()

# NADE simulator
class nade_cr_simulator(sumo_cr_simulator):
    def reset(
        self,
        interactive_scenario_path: str,
        scenario=None,
        planning_problem_set=None,
        planning_problem=None,
        ego_vehicle=None,
        reset_config=None,
        output_file=None,
        episode=1,
        worker_id=0,
        single_input_path = None,
    ):
        
        # replay path
        self.replay_path = single_input_path

        # load the planning problem
        self.planning_problem_set = planning_problem_set
        self.planning_problem = planning_problem

        # load the scenario
        self.sumo_initialization = None
        self.sumocfg_file = os.path.join(
            interactive_scenario_path, f"environment.sumocfg"
        )
        self.sumonet_file = os.path.join(interactive_scenario_path, f"road.net.xml")
        self.cr_map_file = os.path.join(interactive_scenario_path, f"cr_map.xml")
        self.scenario_wrapper = ScenarioWrapper(sumo_cfg_file=self.sumocfg_file)
        self.scenario_wrapper.initialize_sumo(
            scenario=scenario,
            cr_map_file=self.cr_map_file,
            time_step_size=self.sumo_config.dt,
        )

        # create the simulation object
        self.simulation = simulate_scenario_nade(
            self.obs_configs,
            self.sumo_config,
            self.scenario_wrapper,
            planning_problem_set=planning_problem_set,
            planning_problem=planning_problem,
            sumo_initialization=self.sumo_initialization,
            ego_vehicle=ego_vehicle,
            reset_config=reset_config,
            output_file=output_file,
            sumo_net_file=self.sumonet_file,
            episode=episode,
            worker_id=worker_id,
            single_input_path=single_input_path,
        )

        return self.observation()
