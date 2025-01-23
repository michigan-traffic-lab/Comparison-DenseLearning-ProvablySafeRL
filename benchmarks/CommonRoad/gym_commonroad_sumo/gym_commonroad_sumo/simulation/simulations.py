"""
SUMO simulation specific helper methods
"""

import copy
import os
import pickle
import math
from typing import Union, Tuple, Dict, Optional, List, Set
import gym
import xml.etree.ElementTree as ET

import numpy as np
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.scenario.scenario import Scenario
from sumocr.interface.ego_vehicle import EgoVehicle
from gym_commonroad_sumo.simulation.sumo_simulation import SumoSimulation
#from sumocr.maps.sumo_scenario import ScenarioWrapper
from gym_commonroad_sumo.simulation.sumo_scenario import ScenarioWrapper
from sumocr.sumo_config.default import DefaultConfig
#from sumocr.visualization.video import create_video
from gym_commonroad_sumo.simulation.video_render import create_video
from gym_commonroad_sumo.simulation.video_render_for_replay import create_video_for_replay
from commonroad.scenario.trajectory import State

from collections import OrderedDict
from gym_commonroad_sumo.observation.ego_observation import EgoObservation
from gym_commonroad_sumo.observation.goal_observation import GoalObservation
from gym_commonroad_sumo.observation.lanelet_network_observation import LaneletNetworkObservation
from gym_commonroad_sumo.observation.surrounding_observation import SurroundingObservation
from gym_commonroad_sumo.observation.traffic_sign_observation import TrafficSignObservation
from gym_commonroad_sumo.utils.navigator import Navigator
from commonroad_route_planner.route_planner import RoutePlanner
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, Obstacle, StaticObstacle
import commonroad_dc
from commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch import create_collision_object, \
    create_collision_checker
from commonroad_dc.collision.trajectory_queries import trajectory_queries
from commonroad.geometry.shape import Polygon, Rectangle, Circle
import commonroad_dc.pycrcc as pycrcc
from commonroad.scenario.lanelet import Lanelet
from commonroad_dc.pycrccosy import CurvilinearCoordinateSystem
from gym_commonroad_sumo.action.vehicle import Vehicle
from gym_commonroad_sumo.utils.scenario import get_lane_marker

class simulate_scenario(object):
    """
    Simulates an interactive scenario with specified mode

    :param conf: config of the simulation
    :param scenario_wrapper: scenario wrapper used by the Simulator
    :param planning_problem_set: planning problem set of the scenario
    :return: simulated scenario and dictionary with items {planning_problem_id: EgoVehicle}
    """
    def __init__(self, 
                obs_configs: Dict,
                conf: DefaultConfig,
                scenario_wrapper: ScenarioWrapper,
                planning_problem_set: PlanningProblemSet = None,
                planning_problem: PlanningProblem = None,
                sumo_initialization = None,
                ego_vehicle: Vehicle = None,
                reset_config = None,
                output_file = None,
                episode = 1):

        # initialization info
        self._planning_problem_set = planning_problem_set
        self._planning_problem = planning_problem
        self._scenario_id = scenario_wrapper.initial_scenario.scenario_id
        self._scenario = scenario_wrapper.initial_scenario
        self.current_scenario = self.copy_scenario(self._scenario)
        self.ego_vehicle = ego_vehicle

        self.sumo_sim = SumoSimulation()

        # initialize simulation
        self.sumo_sim.initialize(conf, scenario_wrapper, planning_problem_set, sumo_initialization, output_file, episode)
        initial_obstacles = self.sumo_sim._get_cr_obstacles_at_time(0)
        obstacle_dict = dict()
        for o in initial_obstacles:
            obstacle_dict[o.obstacle_id] = o
        self.current_scenario._dynamic_obstacles = obstacle_dict
        self.current_scenario_full = self.copy_scenario(self.current_scenario)

        # initialize basic components
        self.navigator: Navigator = None
        self._continous_collision_check = obs_configs["action_configs"].get("continuous_collision_checking", True)
        self._max_lane_merge_range = obs_configs.get("max_lane_merge_range")
        if planning_problem is not None:
            self._goal_region = self._planning_problem.goal
            self.episode_length = max(s.time_step.end for s in self._goal_region.state_list)
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
        self.lane_circ_sensor_range_radius: float = obs_configs.get("lane_circ_sensor_range_radius", 100)
        self.max_obs_dist = self.lane_circ_sensor_range_radius
        self._road_edge = {
            "left_road_edge_lanelet_id_dict": reset_config["left_road_edge_lanelet_id_dict"],
            "right_road_edge_lanelet_id_dict": reset_config["right_road_edge_lanelet_id_dict"],
            "left_road_edge_dict": reset_config["left_road_edge_dict"],
            "right_road_edge_dict": reset_config["right_road_edge_dict"],
            "boundary_collision_object": reset_config["boundary_collision_object"]}
        
        # initialize observation
        self.observation_space_size: int = None
        self._flatten_observation = obs_configs.get("flatten_observation")
        self.observation_dict = OrderedDict()
        self.observation_history_dict: dict = dict()
        self.observation_space = self._build_observation_space()
        
    @property
    def road_edge(self):
        return self._road_edge
    
    @staticmethod
    def copy_scenario(scenario: Scenario) -> Scenario:
        # copy the scenario while avoiding using pointer and changing the original scenario
        scenario_new = Scenario(dt=scenario.dt, scenario_id=scenario.scenario_id)
        scenario_new.lanelet_network = scenario.lanelet_network
        scenario_new.add_objects(copy.deepcopy(scenario.obstacles))
        return scenario_new

    def _build_observation_space(self) -> Union[gym.spaces.Box, gym.spaces.Dict]:
        """
        builds the observation space dictionary

        :return: the function returns an OrderedDict with the observation spaces
        """
        observation_space_dict = OrderedDict()
        observation_space_dict["distance_goal_long"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["distance_goal_long_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                  dtype=np.float32)
        observation_space_dict["distance_goal_lat"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["distance_goal_lat_advance"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        observation_space_dict["distance_goal_time"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["distance_goal_orientation"] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                                                                 dtype=np.float32)
        observation_space_dict["distance_goal_velocity"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["is_goal_reached"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        observation_space_dict["is_time_out"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        observation_space_dict["v_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["a_ego"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["relative_heading"] = gym.spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32)
        observation_space_dict["global_turn_rate"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["is_friction_violation"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        observation_space_dict["remaining_steps"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["lane_based_v_rel"] = gym.spaces.Box(-np.inf, np.inf, (6,), dtype=np.float32)
        observation_space_dict["lane_based_p_rel"] = gym.spaces.Box(-self.max_obs_dist, self.max_obs_dist, (6,),
                                                                        dtype=np.float32)
        observation_space_dict["is_collision"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        observation_space_dict["lane_change"] = gym.spaces.Box(0, 1, (1,), dtype=np.float32)
        observation_space_dict["is_off_road"] = gym.spaces.Box(0, 1, (1,), dtype=np.int8)
        observation_space_dict["left_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["right_marker_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["left_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["right_road_edge_distance"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        observation_space_dict["lat_offset"] = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

        self.observation_space_dict = observation_space_dict
        if self._flatten_observation:
            lower_bounds, upper_bounds = np.array([]), np.array([])
            for space in observation_space_dict.values():
                lower_bounds = np.concatenate((lower_bounds, space.low))
                upper_bounds = np.concatenate((upper_bounds, space.high))
            self.observation_space_size = lower_bounds.shape[0]
            observation_space = gym.spaces.Box(low=lower_bounds, high=upper_bounds, dtype=np.float64)
        else:
            observation_space = gym.spaces.Dict(self.observation_space_dict)

        return observation_space      
    
    def _create_navigator(self):
        """
        creates and stores the Navigator of the current scenario
        """
        route_planner = RoutePlanner(self._scenario, self._planning_problem,
                                        backend=RoutePlanner.Backend.NETWORKX_REVERSED, log_to_console=False, )
        route_candidates = route_planner.plan_routes()
        route = route_candidates.retrieve_best_route_by_orientation()
        self.navigator = Navigator(route)
    
    @staticmethod
    def compute_convex_hull_circle(radius, previous_position, current_position) -> pycrcc.RectOBB:
        """ Compute obb based on last and current position to
            approximate the area covered by the collision circle between
            the last and current timestep.
        """
        position = (current_position + previous_position) / 2.0
        direction = current_position - previous_position
        direction_length = np.linalg.norm(direction)
        d_normed = direction / direction_length
        orientation = np.arctan2(d_normed[1], d_normed[0])

        return pycrcc.RectOBB(direction_length / 2 + radius, radius, orientation, position[0], position[1])

    @staticmethod
    def create_convex_hull_collision_circle(dynamic_obstacle: DynamicObstacle):
        assert isinstance(dynamic_obstacle.obstacle_shape, Circle)

        initial_time_step = dynamic_obstacle.initial_state.time_step
        tvo = pycrcc.TimeVariantCollisionObject(initial_time_step)
        if dynamic_obstacle.prediction is not None:
            for time_step in range(initial_time_step, dynamic_obstacle.prediction.final_time_step):
                previous_state = dynamic_obstacle.state_at_time(time_step)
                state = dynamic_obstacle.state_at_time(time_step + 1)
                convex_obb = simulate_scenario.compute_convex_hull_circle(
                    dynamic_obstacle.obstacle_shape.radius, previous_state.position, state.position)
                tvo.append_obstacle(convex_obb)
        else: # SUMO scenario
            tvo.append_obstacle(create_collision_object(dynamic_obstacle))

        return tvo

    @staticmethod
    def create_collision_checker_scenario(scenario: Scenario, params=None, collision_object_func=None, continous_collision_check=True):
        # create collision checker for the current scenario
        if not continous_collision_check:
            return create_collision_checker(scenario)
        cc = pycrcc.CollisionChecker()
        for co in scenario.dynamic_obstacles:
            if isinstance(co.obstacle_shape, Rectangle):
                collision_object = create_collision_object(co, params, collision_object_func)
                if co.prediction is not None:
                # TODO: remove if when https://gitlab.lrz.de/cps/commonroad-drivability-checker/-/issues/16 is fixed
                    collision_object, err = trajectory_queries.trajectory_preprocess_obb_sum(collision_object)
                    if err:
                        raise Exception("<ObservationCollector.create_collision_checker_scenario> Error create convex hull")
            elif isinstance(co.obstacle_shape, Circle):
                collision_object = simulate_scenario.create_convex_hull_collision_circle(co)
            else:
                raise NotImplementedError(f"Unsupported shape for convex hull collision object: {co.obstacle_shape}")
            cc.add_collision_object(collision_object)

        shape_group = pycrcc.ShapeGroup()
        for co in scenario.static_obstacles:
            collision_object = commonroad_dc.collision.collision_detection.pycrcc_collision_dispatch. \
                create_collision_object(co, params, collision_object_func)
            if isinstance(collision_object, pycrcc.ShapeGroup):
                for shape in collision_object.unpack():
                    shape_group.add_shape(shape)
            else:
                shape_group.add_shape(collision_object)
        cc.add_collision_object(shape_group)
        return cc

    def _update_collision_checker(self):
        # update collision checker by current obstacles
        self.current_scenario_new = self.copy_scenario(self.current_scenario)
        for i in range(len(self.current_scenario_new.dynamic_obstacles)):
            self.current_scenario_new.dynamic_obstacles[i].initial_state.time_step = self.sumo_sim.current_time_step

        self._collision_checker = self.create_collision_checker_scenario(
            self.current_scenario_new, continous_collision_check=self._continous_collision_check)
        
    def get_collision_checker(self):
        # return collision checker
        return self._collision_checker
    
    def _get_lanelet_polygons(self):
        """
        returns lanelet_polygons and the shape group of the polygons
        """
        lanelet_polygons = [(lanelet.lanelet_id, lanelet.convert_to_polygon()) for lanelet in
                                self._scenario.lanelet_network.lanelets]
        lanelet_polygons_sg = pycrcc.ShapeGroup()
        for l_id, poly in lanelet_polygons:
            lanelet_polygons_sg.add_shape(create_collision_object(poly))
        self.lanelet_polygons = lanelet_polygons
        self.lanelet_polygons_sg = lanelet_polygons_sg
    
    def sorted_lanelets_by_state(self, state: State,) -> List[int]:
        """
        Returns the sorted list of lanelet ids which correspond to a given state
        """
        # sort the lanelet by relative orientation
        return Navigator.sorted_lanelet_ids(
            self._related_lanelets_by_state(state),
            state.orientation if hasattr(state, "orientation") else np.arctan2(state.velocity_y, state.velocity),
            state.position, self._scenario, )

    def _related_lanelets_by_state(self, state: State) -> List[int]:
        """
        Get the lanelet of a state. Include all the lanelet that the state is in.
        """
        # output list
        res = list()

        # look at each lanelet
        point_list = [state.position]

        point_sg = pycrcc.ShapeGroup()
        for el in point_list:
            point_sg.add_shape(pycrcc.Point(el[0], el[1]))

        lanelet_polygon_ids = point_sg.overlap_map(self.lanelet_polygons_sg)

        for lanelet_id_list in lanelet_polygon_ids.values():
            for lanelet_id in lanelet_id_list:
                res.append(self.lanelet_polygons[lanelet_id][0])

        return res
    
    def get_local_curvi_cosy(self) -> Tuple[CurvilinearCoordinateSystem, Lanelet]:
        """
        At every time step, update the local curvilinear coordinate system from the dict.
        :param ego_vehicle_lanelet_id: The lanelet id where the ego vehicle is on
        :return: Curvilinear coordinate system of the merged lanelets
        """
        ref_path, ref_merged_lanelet, curvi_cosy = (None, None, None)

        if curvi_cosy is None:
            for lanelet in self._scenario.lanelet_network.lanelets:  # iterate in all lanelet in this scenario
                if lanelet.lanelet_id == self.ego_lanelet_id and (
                        not lanelet.predecessor and not lanelet.successor):  # the lanelet is a lane itself
                    ref_path = lanelet.center_vertices
                    ref_merged_lanelet = lanelet
                elif not lanelet.predecessor:  # the lanelet is the start of a lane, the lane can be created from here
                    # TODO: cache merged lanelets in pickle or dict
                    merged_lanelet_list, sub_lanelet_ids_list = Lanelet.all_lanelets_by_merging_successors_from_lanelet(
                        lanelet, self._scenario.lanelet_network, self._max_lane_merge_range)
                    for merged_lanelet, sub_lanelet_ids in zip(merged_lanelet_list, sub_lanelet_ids_list):
                        if self.ego_lanelet_id in sub_lanelet_ids:
                            ref_path = merged_lanelet.center_vertices
                            ref_merged_lanelet = merged_lanelet
                            break
            curvi_cosy = Navigator.create_coordinate_system_from_polyline(ref_path)

        return curvi_cosy, ref_merged_lanelet

    def _update_ego_lanelet_and_local_ccosy(self):
        # update ego lanelet id
        self._ego_lanelet_ids = self.sorted_lanelets_by_state(self.ego_vehicle.state)
        if len(self._ego_lanelet_ids) == 0:
            ego_lanelet_id = self.ego_lanelet_id
        else:
            self.ego_lanelet_id = ego_lanelet_id = self._ego_lanelet_ids[0]

        self.ego_lanelet = self._scenario.lanelet_network.find_lanelet_by_id(ego_lanelet_id)

        # update local ccosy
        self.local_ccosy, self._local_merged_lanelet = self.get_local_curvi_cosy()

    def get_long_lat_distance_to_goal(self, position: np.array) -> Tuple[float, float]:
        """
        Get longitudinal and lateral distances to the goal over the planned route
        """
        try:
            return self.navigator.get_long_lat_distance_to_goal(position)
        except ValueError:
            return np.nan, np.nan
        
    def _get_long_lat_distance_advance_to_goal(self, distance_goal_long: float,
                                               distance_goal_lat: float) -> Tuple[float, float]:
        """
        Get longitudinal and lateral distances to the goal over the planned route
        """
        distance_goal_long_advance = abs(self.observation_history_dict["distance_goal_long"]) - \
            abs(distance_goal_long) if "distance_goal_long" in self.observation_history_dict else 0.0
        distance_goal_lat_advance = abs(self.observation_history_dict["distance_goal_lat"]) - \
            abs(distance_goal_lat) if "distance_goal_lat" in self.observation_history_dict else 0.0

        return distance_goal_long_advance, distance_goal_lat_advance
    
    def _check_goal_reached(self) -> bool:
        """
        Check if goal is reached by ego vehicle.
        """
        if self.ego_vehicle.state.time_step == 0:
            return False
        return self._planning_problem.goal.is_reached(self.ego_vehicle.state)
    
    def get_nearby_lanelet_id(self) -> Tuple[dict, set]:
        """
        Get ids of nearby lanelets, e.g. lanelets that are successors, predecessors, left, or right of the
        `ego_vehicle_lanelet`
        """
        keys = {"ego", "left", "right", "ego_other", "left_other", "right_other", "ego_all", "left_all", "right_all", }
        lanelet_dict = {key: set() for key in keys}
        ego_vehicle_lanelet_id = self.ego_lanelet.lanelet_id
        lanelet_dict["ego"].add(ego_vehicle_lanelet_id)  # Add ego lanelet

        if self.ego_lanelet.adj_right_same_direction:
            # Get adj right lanelet with same direction
            lanelet_dict["right"].add(self.ego_lanelet.adj_right)
        if self.ego_lanelet.adj_left_same_direction:
            # Get adj left lanelet with same direction
            lanelet_dict["left"].add(self.ego_lanelet.adj_left)

        for ego_lanelet_id in lanelet_dict["ego"]:
            lanelet_dict["ego_other"].update({ego_lanelet_id})
        for left_lanelet_id in lanelet_dict["left"]:
            lanelet_dict["left_other"].update({left_lanelet_id})
        for right_lanelet_id in lanelet_dict["right"]:
            lanelet_dict["right_other"].update({right_lanelet_id})

        lanelet_dict["ego_all"] = set().union(set(lanelet_dict["ego"]), set(lanelet_dict["ego_other"]))
        lanelet_dict["left_all"] = set().union(set(lanelet_dict["left"]), set(lanelet_dict["left_other"]))
        lanelet_dict["right_all"] = set().union(set(lanelet_dict["right"]), set(lanelet_dict["right_other"]))

        all_lanelets_set = set().union(lanelet_dict["ego_all"], lanelet_dict["left_all"], lanelet_dict["right_all"])

        return lanelet_dict, all_lanelets_set
    
    def _get_occupied_lanelet_id(self, obstacle_lanelet_ids: List[int], obstacle_state: State) \
            -> Union[None, int]:
        """
        gets most relevant lanelet id from obstacle_lanelet_ids for an obstacle that occupies multiple lanelets
        """
        if len(obstacle_lanelet_ids) > 1:
            # select the most relevant lanelet
            return Navigator.sorted_lanelet_ids(
                obstacle_lanelet_ids, obstacle_state.orientation, obstacle_state.position, self._scenario)[0]
        elif len(obstacle_lanelet_ids) == 1:
            return obstacle_lanelet_ids[0]
        else:
            return None

    def _get_obstacles_in_surrounding_area(self, surrounding_area: pycrcc.Shape) \
            -> Tuple[List[int], List[State], List[Obstacle]]:
        """
        Get the states and lanelet ids of all obstacles within the range of surrounding area of ego vehicle.
        :return: List of lanelet ids of obstacles, list of states obstacles
        """
        lanelet_ids, obstacle_states, obstacles = [], [], []
        dyn_obstacles, static_obstacles = self.current_scenario.dynamic_obstacles, self.current_scenario.static_obstacles

        # iterate over all dynamic obstacles
        for o in dyn_obstacles:
            # obstacle only has initial state in interactive scenarios
            obstacle_state = o.state_at_time(0)
            obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
            if surrounding_area.collide(obstacle_point):
                o_center_lanelet_ids = self.sorted_lanelets_by_state(obstacle_state)
                o_lanelet_id = self._get_occupied_lanelet_id(o_center_lanelet_ids, obstacle_state)
                lanelet_ids.append(o_lanelet_id)
                obstacle_states.append(obstacle_state)
                obstacles.append(o)

        # iterate over all static obstacles
        for o in static_obstacles:
            obstacle_state = o.initial_state
            obstacle_point = pycrcc.Point(obstacle_state.position[0], obstacle_state.position[1])
            if surrounding_area.collide(obstacle_point):
                obstacle_lanelet_ids = list(o.initial_center_lanelet_ids)
                lanelet_id = self._get_occupied_lanelet_id(self._scenario, obstacle_lanelet_ids, obstacle_state)
                lanelet_ids.append(lanelet_id)
                obstacle_states.append(obstacle_state)
                obstacles.append(o)

        return lanelet_ids, obstacle_states, obstacles
    
    @staticmethod
    def _filter_obstacles_in_adj_lanelet(lanelet_ids: List[int], states: List[State], obstacles: List[Obstacle],
                                         all_lanelets_set: Set[int]) -> Tuple[List[int], List[State], List[Obstacle]]:
        """
        filters out obstacles states and their corresponding lanelet id
        where the lanelet id is not in the all_lanelets_set
        """
        adj_obstacle_states, obstacle_lanelet, adj_obstacles = [], [], []
        for lanelet_id, state, obstacle in zip(lanelet_ids, states, obstacles):
            if lanelet_id in all_lanelets_set:  # Check if the obstacle is in adj lanelets
                obstacle_lanelet.append(lanelet_id)
                adj_obstacle_states.append(state)
                adj_obstacles.append(obstacle)

        return obstacle_lanelet, adj_obstacle_states, adj_obstacles

    def _get_ego_obstacle_distance(self, obstacle_state: State, ego_curvi: Tuple[float, float]) \
            -> Tuple[float, int]:
        """
        Get the distance between the ego_vehicle and an obstacle
        """
        ego_curvi_long_position, _ = ego_curvi
        try:
            o_curvi_long_position, _ = self.local_ccosy.convert_to_curvilinear_coords(obstacle_state.position[0],
                                                                                       obstacle_state.position[1])
        except ValueError:
            # the position is out of project area of curvilinear coordinate system
            o_curvi_long_position = ego_curvi_long_position + self.max_obs_dist
        distance_sign = np.sign(ego_curvi_long_position - o_curvi_long_position)
        dist_abs = np.abs(ego_curvi_long_position - o_curvi_long_position)

        return dist_abs, distance_sign
    
    @staticmethod
    def get_rel_v_p_follow_leading(distance_sign: int, distance_abs: float, p_rel_follow: float, p_rel_lead: float,
                                   v_rel_follow: float, v_rel_lead: float, obs_state: State, obstacle: Obstacle,
                                   ego_state: State, o_follow: State, o_lead: State, obstacle_follow: Obstacle,
                                   obstacle_lead: Obstacle) -> \
            Tuple[float, float, State, Obstacle, float, float, State, Obstacle]:
        """
        calculates the relative velocity of leading and following obstacles to the ego vehicle
        """
        if isinstance(obstacle, StaticObstacle):
            obs_state.velocity = 0.
        if distance_sign == 1 and distance_abs < p_rel_follow:
            # following vehicle, distance is smaller
            ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else np.arctan2(
                ego_state.velocity_y, ego_state.velocity)
            delta_orientation = obs_state.orientation - ego_state_orientation
            v_rel_follow = ego_state.velocity - obs_state.velocity * np.cos(delta_orientation)
            p_rel_follow = distance_abs
            o_follow = obs_state
            obstacle_follow = obstacle
        elif distance_sign != 1 and distance_abs < p_rel_lead:
            # leading vehicle, distance is smaller
            ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else np.arctan2(
                ego_state.velocity_y, ego_state.velocity)
            delta_orientation = obs_state.orientation - ego_state_orientation
            v_rel_lead = obs_state.velocity * np.cos(delta_orientation) - ego_state.velocity
            p_rel_lead = distance_abs
            o_lead = obs_state
            obstacle_lead = obstacle

        return v_rel_follow, p_rel_follow, o_follow, obstacle_follow, v_rel_lead, p_rel_lead, o_lead, obstacle_lead

    def _get_rel_v_p_lane_based(
            self, obstacles_lanelet_ids: List[int], obstacle_states: List[State], lanelet_dict: Dict[str, List[int]],
            adj_obstacles: List[Obstacle]) -> Tuple[List[float], List[float], List[State], List[Obstacle], np.array]:
        """
        Get the relative velocity and position of obstacles in adj left, adj right and ego lanelet.
        In each lanelet, compute only the nearest leading and following obstacles.
        """
        # Initialize dummy values, in case no obstacles are present
        v_rel_left_follow, v_rel_same_follow, v_rel_right_follow, v_rel_left_lead, v_rel_same_lead, \
        v_rel_right_lead = [0.0] * 6

        p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead, \
        p_rel_right_lead = [self.max_obs_dist] * 6

        try:
            ego_vehicle_long_position, ego_vehicle_lat_position = self.local_ccosy.convert_to_curvilinear_coords(
                self.ego_vehicle.state.position[0], self.ego_vehicle.state.position[1])

            o_left_follow, o_left_lead, o_right_follow, o_right_lead, o_same_follow, o_same_lead = [None] * 6
            obstacle_left_follow, obstacle_left_lead, obstacle_right_follow, obstacle_right_lead, \
            obstacle_same_follow, obstacle_same_lead = [None] * 6

            for o_state, o_lanelet_id, obstacle in zip(obstacle_states, obstacles_lanelet_ids, adj_obstacles):

                distance_abs, distance_sign = self._get_ego_obstacle_distance(o_state,
                                                                              (ego_vehicle_long_position,
                                                                               ego_vehicle_lat_position))

                if o_lanelet_id in lanelet_dict["ego_all"]:  # ego lanelet
                    v_rel_same_follow, p_rel_same_follow, o_same_follow, obstacle_same_follow, \
                    v_rel_same_lead, p_rel_same_lead, o_same_lead, obstacle_same_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_same_follow, p_rel_same_lead, v_rel_same_follow,
                            v_rel_same_lead, o_state, obstacle, self.ego_vehicle.state, o_same_follow, o_same_lead,
                            obstacle_same_follow, obstacle_same_lead)

                if o_lanelet_id in lanelet_dict["right_all"]:  # right lanelet
                    v_rel_right_follow, p_rel_right_follow, o_right_follow, obstacle_right_follow, \
                    v_rel_right_lead, p_rel_right_lead, o_right_lead, obstacle_right_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_right_follow, p_rel_right_lead, v_rel_right_follow,
                            v_rel_right_lead, o_state, obstacle, self.ego_vehicle.state, o_right_follow, o_right_lead,
                            obstacle_right_follow, obstacle_right_lead)

                if o_lanelet_id in lanelet_dict["left_all"]:  # left lanelet
                    v_rel_left_follow, p_rel_left_follow, o_left_follow, obstacle_left_follow, \
                    v_rel_left_lead, p_rel_left_lead, o_left_lead, obstacle_left_lead = \
                        self.get_rel_v_p_follow_leading(
                            distance_sign, distance_abs, p_rel_left_follow, p_rel_left_lead, v_rel_left_follow,
                            v_rel_left_lead, o_state, obstacle, self.ego_vehicle.state, o_left_follow, o_left_lead,
                            obstacle_left_follow, obstacle_left_lead)

            detected_states = [o_left_follow, o_same_follow, o_right_follow, o_left_lead, o_same_lead, o_right_lead]
            detected_obstacles = [obstacle_left_follow, obstacle_same_follow, obstacle_right_follow,
                                  obstacle_left_lead, obstacle_same_lead, obstacle_right_lead]
        except ValueError:
            detected_states = [None] * 6
            detected_obstacles = [None] * 6
            ego_vehicle_lat_position = None

        v_rel = [v_rel_left_follow, v_rel_same_follow, v_rel_right_follow, v_rel_left_lead, v_rel_same_lead,
                 v_rel_right_lead]
        p_rel = [p_rel_left_follow, p_rel_same_follow, p_rel_right_follow, p_rel_left_lead, p_rel_same_lead,
                 p_rel_right_lead]

        return v_rel, p_rel, detected_states, detected_obstacles, ego_vehicle_lat_position

    def _get_surrounding_obstacles_lane_based(self, surrounding_area: Union[pycrcc.RectOBB, pycrcc.Circle]) \
            -> Tuple[np.array, List[State], List[Obstacle]]:
        # get the relative velocity and distance between ego vehicle and obstacles in the surrounding area
       
        lanelet_ids, obstacle_states, obstacles = self._get_obstacles_in_surrounding_area(surrounding_area)
        obstacle_lanelet, adj_obstacle_states, adj_obstacles = \
            self._filter_obstacles_in_adj_lanelet(lanelet_ids, obstacle_states, obstacles, self.all_lanelets_set)
        rel_vel, rel_pos, detected_states, detected_obstacles, ego_vehicle_lat_position = \
            self._get_rel_v_p_lane_based(obstacle_lanelet, adj_obstacle_states, self.lanelet_dict, adj_obstacles)

        self.observation_dict["lane_based_v_rel"] = np.array(rel_vel)
        self.observation_dict["lane_based_p_rel"] = np.array(rel_pos)

        self.conflict_obstacles_information = [state for state in obstacle_states if state not in detected_states]

        return ego_vehicle_lat_position, detected_states, detected_obstacles
    
    def _detect_lane_change(self) -> None:
        # check if ego vehicle is changing lane
        self.observation_dict["lane_change"] = np.array([0.0])
        for lanelet_id in self._ego_lanelet_ids:
            if lanelet_id not in self.lanelet_dict["ego_all"]:
                self.observation_dict["lane_change"] = np.array([1.0])
                return

    def _check_collision(self) -> bool:
        # check if ego vehicle is in collision
        collision_ego_vehicle = self.ego_vehicle.collision_object
        is_collision = self._collision_checker.collide(collision_ego_vehicle)
        return is_collision
    
    def _check_is_off_road(self) -> bool:
        # check if ego vehicle is off road
        collision_ego_vehicle = self.ego_vehicle.collision_object
        is_off_road = collision_ego_vehicle.collide(self._road_edge["boundary_collision_object"])
        return is_off_road
    
    def _get_road_edge(self):
        """
        Get the left and right road edge of ego vehicle lanelet.
        """
        left_most_lanelet_id = self._road_edge["left_road_edge_lanelet_id_dict"][self.ego_lanelet_id]
        right_most_lanelet_id = self._road_edge["right_road_edge_lanelet_id_dict"][self.ego_lanelet_id]
        left_road_edge = self._road_edge["left_road_edge_dict"][left_most_lanelet_id]
        right_road_edge = self._road_edge["right_road_edge_dict"][right_most_lanelet_id]
        return left_road_edge, right_road_edge

    def _get_distance_to_marker_and_road_edge(self):
        # get the ego vehicle's distance to the lane marker and road edge
        left_marker_line, right_marker_line = get_lane_marker(self.ego_lanelet)
        current_left_road_edge, current_right_road_edge = self._get_road_edge()
        (left_marker_distance, right_marker_distance, left_road_edge_distance, right_road_edge_distance) \
            = LaneletNetworkObservation.get_distance_to_marker_and_road_edge(self.ego_vehicle.state, left_marker_line,
                                                                             right_marker_line, current_left_road_edge,
                                                                             current_right_road_edge)
        self.observation_dict["left_marker_distance"] = np.array([left_marker_distance])
        self.observation_dict["right_marker_distance"] = np.array([right_marker_distance])
        self.observation_dict["left_road_edge_distance"] = np.array([left_road_edge_distance])
        self.observation_dict["right_road_edge_distance"] = np.array([right_road_edge_distance])

    def _get_lat_offset(self):
        # return the lateral offset of the ego vehicle from the center of the lane
        lat_offset = LaneletNetworkObservation.get_relative_offset(self.local_ccosy, self.ego_vehicle.state.position)
        if np.isnan(lat_offset) and "lat_offset" in self.observation_history_dict.keys():
            lat_offset = self.observation_history_dict["lat_offset"][-1]
        self.observation_history_dict["lat_offset"] = np.array([lat_offset])
        self.observation_dict["lat_offset"] = np.array([lat_offset])

    def observe_ego(self):
        # observe the ego vehicle's velocity, acceleration, heading, yaw rate, friction violation, and remaining steps
        ego_state = self.ego_vehicle.state
        self.observation_dict["v_ego"] = np.array([ego_state.velocity])
        self.observation_dict["a_ego"] = np.array([ego_state.acceleration])
        relative_heading = EgoObservation.get_lane_relative_heading(ego_state, self.ego_lanelet)
        self.observation_dict["relative_heading"] = relative_heading
        self.observation_dict["global_turn_rate"] = np.array([ego_state.yaw_rate])
        is_friction_violation = self.ego_vehicle.violate_friction
        self.observation_dict["is_friction_violation"] = np.array([is_friction_violation])
        self.observation_dict["remaining_steps"] = np.array([self.episode_length - ego_state.time_step])

    def observe_goal(self):
        # observe the gap between ego vehicle and the goal

        ego_state = self.ego_vehicle.state
        # observe distance goal long and lat and also the advance versions
        distance_goal_long, distance_goal_lat = self.get_long_lat_distance_to_goal(ego_state.position)
        self.observation_dict["distance_goal_long"] = np.array([distance_goal_long])
        self.observation_dict["distance_goal_lat"] = np.array([distance_goal_lat])
        if distance_goal_long is np.nan:
            distance_goal_long = self.observation_history_dict.get("distance_goal_long", 1e4)
            distance_goal_lat = self.observation_history_dict.get("distance_goal_lat", 1e4)
        (distance_goal_long_advance, distance_goal_lat_advance) = \
            self._get_long_lat_distance_advance_to_goal(distance_goal_long, distance_goal_lat)
        self.observation_history_dict["distance_goal_long_advance"] = distance_goal_long_advance
        self.observation_history_dict["distance_goal_lat_advance"] = distance_goal_lat_advance
        self.observation_history_dict["distance_goal_long"] = distance_goal_long
        self.observation_history_dict["distance_goal_lat"] = distance_goal_lat
        self.observation_dict["distance_goal_long_advance"] = np.array([distance_goal_long_advance])
        self.observation_dict["distance_goal_lat_advance"] = np.array([distance_goal_lat_advance])

        # observe the time to the goal state
        distance_goal_time = GoalObservation._get_goal_time_distance(ego_state.time_step, self._planning_problem.goal)
        self.observation_dict["distance_goal_time"] = np.array([distance_goal_time])
        # observe distance to the goal orientation from ego
        ego_state_orientation = ego_state.orientation if hasattr(ego_state, "orientation") else \
            np.arctan2(ego_state.velocity_y, ego_state.velocity)
        distance_goal_orientation = GoalObservation._get_goal_orientation_distance(ego_state_orientation, self._planning_problem.goal)
        self.observation_dict["distance_goal_orientation"] = np.array([distance_goal_orientation])
        # observe difference between ego velocity and goal velocity
        distance_goal_velocity = GoalObservation._get_goal_velocity_distance(ego_state.velocity, self._planning_problem.goal)
        self.observation_dict["distance_goal_velocity"] = np.array([distance_goal_velocity])
        # Check if the ego vehicle has reached the goal
        is_goal_reached = self._check_goal_reached()
        self.observation_dict["is_goal_reached"] = np.array([is_goal_reached])
        # Check if maximum episode length exceeded and check if ego vehicle reaches the end of the road
        is_time_out = GoalObservation._check_is_time_out(ego_state, self._planning_problem.goal, is_goal_reached, self.episode_length)
        if not is_time_out:
            # check if ego vehicle reaches the end of the road
            if not self.local_ccosy.cartesian_point_inside_projection_domain(ego_state.position[0], ego_state.position[1]):
                is_time_out=True
        self.observation_dict["is_time_out"] = np.array([is_time_out])

    def observe_surrounding(self):
        # get the surrounding obstacles' states and check collision
        self.lanelet_dict, self.all_lanelets_set = self.get_nearby_lanelet_id()
        self._surrounding_area = pycrcc.Circle(self.lane_circ_sensor_range_radius,
                                                self.ego_vehicle.state.position[0],
                                                self.ego_vehicle.state.position[1])
        ego_vehicle_lat_position, self._detected_obstacle_states, self.detected_obstacles = \
            self._get_surrounding_obstacles_lane_based(self._surrounding_area)
        self._detect_lane_change()

        # check if ego vehicle is in collision
        is_collision = self._check_collision()
        self.observation_dict["is_collision"] = np.array([is_collision])

    def observe_lanelet(self):
        # check if ego vehicle is off road
        is_off_road = self._check_is_off_road()
        self.observation_dict["is_off_road"] = np.array([is_off_road])
        self._get_distance_to_marker_and_road_edge()
        self._get_lat_offset()

    def observe_traffic_sign(self):
        pass

    def run_simulation_one_step(self, ego_vehicle: Vehicle):
        # run one step of the simulation

        self.ego_vehicle = copy.deepcopy(ego_vehicle)
        ego_state = copy.deepcopy(self.ego_vehicle.state)
        ego_state.time_step = 1

        # update ego vehicle info
        ego_vehicles = copy.deepcopy(self.sumo_sim.ego_vehicles)
        for idx in [id for id in ego_vehicles.keys()]:
            trajectory_ego = [ego_state]
            ego_vehicles[idx].set_planned_trajectory(trajectory_ego)

        # step sumo environment
        self.sumo_sim.ego_vehicles = ego_vehicles
        self.sumo_sim.simulate_step()
        self.ego_vehicle.state.time_step = self.sumo_sim.current_time_step

        # update all the info for current time step
        self.update()
    
    def update(self):
        # retrieve the current scenario
        self.current_scenario = self.sumo_sim.commonroad_scenario_at_time_step(self.sumo_sim.current_time_step)
        self.current_scenario_full = self.sumo_sim.commonroad_scenarios_all_time_steps() # scenario with correct obstacle time step, used for collision checking

        # update basic info
        self._update_ego_lanelet_and_local_ccosy()
        self._update_collision_checker()

        # update observation dict
        self.observation_dict = OrderedDict()
        self.observe_ego()
        self.observe_goal()
        self.observe_surrounding()
        self.observe_lanelet()
        self.observe_traffic_sign()
        self.observation_dict = OrderedDict((k, self.observation_dict[k]) for k in self.observation_space_dict.keys())

    def observation(self):
        # return the current observation from observation dict
        if self._flatten_observation:
            observation_vector = np.zeros(self.observation_space.shape)
            index = 0
            for k in self.observation_dict.keys():
                size = np.prod(self.observation_dict[k].shape)
                observation_vector[index: index + size] = self.observation_dict[k].flat
                index += size
            return observation_vector
        else:
            return self.observation_dict

    def get_results(self):
        # retrieve the simulated scenario in CR format
        simulated_scenario = self.sumo_sim.commonroad_scenarios_all_time_steps()

        # stop the simulation
        self.sumo_sim.stop()
        ego_vehicles = {list(self._planning_problem_set.planning_problem_dict.keys())[0]:
                            ego_v for _, ego_v in self.sumo_sim.ego_vehicles.items()}
        simulated_scenario.scenario_id = self._scenario_id

        return simulated_scenario, ego_vehicles

class sumo_cr_simulator(object):
    """
    Simulates an interactive scenario with an outside motion planner for ego vehicle
    """
    def __init__(self, obs_configs: Dict, sumo_config):
        self.simulation = None
        self.obs_configs = obs_configs
        self.sumo_config = sumo_config
        
    def reset(self, 
            interactive_scenario_path: str,
            scenario = None,
            planning_problem_set = None,
            planning_problem = None,
            ego_vehicle = None,
            reset_config = None,
            output_file = None,
            episode = 1,
            worker_id = 0,
            single_input_path = None,):
        
        # replay path
        self.replay_path = single_input_path

        # load the planning problem
        self.planning_problem_set = planning_problem_set
        self.planning_problem = planning_problem

        # load the sumo initialization
        self.sumo_initialization = self.load_sumo_initialization(interactive_scenario_path)

        # load the scenario
        self.sumocfg_file = os.path.join(interactive_scenario_path, f"environment.sumocfg")
        self.cr_map_file = os.path.join(interactive_scenario_path, f"cr_map.xml")
        self.scenario_wrapper = ScenarioWrapper(sumo_cfg_file=self.sumocfg_file)
        self.scenario_wrapper.initialize_sumo(scenario=scenario, cr_map_file=self.cr_map_file, 
                                              time_step_size=self.sumo_config.dt)

        # create the simulation object
        self.simulation = simulate_scenario(self.obs_configs, self.sumo_config,
                            self.scenario_wrapper,
                            planning_problem_set=planning_problem_set,
                            planning_problem=planning_problem,
                            sumo_initialization=self.sumo_initialization,
                            ego_vehicle=ego_vehicle,
                            reset_config=reset_config,
                            output_file=output_file,
                            episode=episode,)

        return self.observation()
        
    def step(self, ego_vehicle: Vehicle):
        # run one step of the sumo simulation
        self.simulation.run_simulation_one_step(ego_vehicle)

    def observation(self):
        # return the current observation of the sumo simulation
        return self.simulation.observation()
    
    def observation_space(self):
        # return the observation space of the sumo simulation
        return self.simulation.observation_space
    
    def observation_dict(self):
        # return the current observation of the sumo simulation
        return self.simulation.observation_dict

    def current_step(self):
        # return the current time step of the sumo simulation
        return self.simulation.sumo_sim.current_time_step if self.simulation is not None else 0

    def get_results(self):
        # retrieve the simulated scenario in CR format
        scenario_with_planner, ego_vehicles = self.simulation.get_results()
        return scenario_with_planner, ego_vehicles

    def load_sumo_initialization(self, interactive_scenario_path: str):
        # load sumo initialization from xml file

        init_tree = ET.parse(os.path.join(interactive_scenario_path, "initialization.xml"))
        init_root = init_tree.getroot()
        vehicles = init_root.findall('vehicle')
        return vehicles

    def create_video_for_simulation(self, scenario: Scenario, 
                                output_folder_path: str,
                                ego_vehicles: Optional[Dict[int, EgoVehicle]],
                                filename: str, 
                                action_dict = None,
                                planning_problem_set: PlanningProblemSet = None,
                                follow_ego: bool = True):
        
        if planning_problem_set is None:
            planning_problem_set = self.planning_problem_set

        """Creates the gif/mp4 animation for the simulation result."""
        if not output_folder_path:
            print("Output folder not specified, skipping mp4 generation.")
            return
        else:
            os.makedirs(output_folder_path, exist_ok=True)

        if self.replay_path is not None:
            with open(os.path.join(self.replay_path, 'simulation_info.pkl'), 'rb') as f:
                simulation_info = pickle.load(f)
            ego_vehicles_unsafe = simulation_info['ego_vehicles']
            ego_vehicles[2] = ego_vehicles_unsafe[1]
            action_dict2 = simulation_info['action_dict']
        else:
            action_dict2 = None

        # create gif/mp4 animation
        if self.replay_path is not None:
            create_video_for_replay(scenario,
                        output_folder_path,
                        planning_problem_set=planning_problem_set,
                        trajectory_pred=ego_vehicles,
                        follow_ego=follow_ego,
                        action_dict=action_dict,
                        action_dict2=action_dict2,
                        filename=filename,
                        file_type='mp4')
        else:
            print(scenario)
            create_video(scenario,
                        output_folder_path,
                        planning_problem_set=planning_problem_set,
                        trajectory_pred=ego_vehicles,
                        follow_ego=follow_ego,
                        action_dict=action_dict,
                        filename=filename,
                        file_type='mp4')
