"""
Module for the CommonRoad Gym environment
"""
import os
import pathlib
import copy
import gym
import yaml
import pickle
import random
import warnings
import numpy as np

from typing import Tuple, Union

# import from commonroad-drivability-checker
from commonroad.geometry.shape import Rectangle, ShapeGroup
from commonroad.prediction.prediction import Occupancy

# import from commonroad-io
from commonroad.scenario.scenario import ScenarioID, Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.obstacle import DynamicObstacle, ObstacleType
from commonroad_rl.gym_commonroad.action import Vehicle
from commonroad_dc import pycrcc

# import from commonroad-rl
from gym_commonroad_sumo.observation import ObservationCollector
from gym_commonroad_sumo.utils.scenario_io import restore_scenario
from gym_commonroad_sumo.utils.scenario import parse_map_name
from gym_commonroad_sumo.action import action_constructor
from gym_commonroad_sumo.reward import reward_constructor
from gym_commonroad_sumo.reward.reward import Reward
from gym_commonroad_sumo.reward.termination import Termination
from gym_commonroad_sumo.utils.collision_type_checker import check_collision_type

from sumocr.visualization.video import create_video
from gym_commonroad_sumo.simulation.simulations import sumo_cr_simulator, simulate_scenario
from gym_commonroad_sumo.file_reader import CommonRoadFileReader
from commonroad_rl.tools.pickle_scenario.preprocessing import generate_reset_config

class CommonroadEnv_SUMO(gym.Env):
    """
    Description:
        This environment simulates the ego vehicle in a traffic scenario using commonroad environment. The task of
        the ego vehicle is to reach the predefined goal without going off-road, collision with other vehicles, and
        finish the task in specific time frame. Please consult `commonroad_rl/gym_commonroad/README.md` for details.
    """

    # For the current configuration check the ./configs.yaml file
    def __init__(
            self,
            test_reset_config_path=None,
            output_path=None,
            config_file=None,
            sumo_config=None,
            planner_dt=0.4,
            **kwargs,
    ) -> None:
        """
        Initialize environment, set scenario and planning problem.
        """
        print("Initialization started")

        self.planner_dt = planner_dt

        # Default configuration
        if isinstance(config_file, (str, pathlib.Path)):
            with pathlib.Path(config_file).open("r") as config_file:
                config = yaml.safe_load(config_file)

        # Assume default environment configurations
        self.configs = config.get("env_configs", config)

        # Overwrite environment configurations if specified
        if kwargs is not None:
            for k, v in kwargs.items():
                assert k in self.configs, f"Configuration item not supported: {k}"
                # TODO: update only one term in configs
                if isinstance(v, dict):
                    self.configs[k].update(v)
                else:
                    self.configs.update({k: v})

        # Make environment configurations as attributes
        self.vehicle_params: dict = self.configs["vehicle_params"]
        self.action_configs: dict = self.configs["action_configs"]
        self.reward_type: str = self.configs["reward_type"]

        # change configurations when using point mass vehicle model
        if self.vehicle_params["vehicle_model"] == 0:
            self.observe_heading = False
            self.observe_steering_angle = False
            self.observe_global_turn_rate = False
            self.observe_distance_goal_long_lane = False

        # Flags for collision type checker evaluation
        self.check_collision_type = self.configs["check_collision_type"]
        self.lane_change_time_threshold = self.configs["lane_change_time_threshold"]

        # Load scenarios and problems
        self.all_problem_dict = dict()
        self.planning_problem_set = []

        # Load reset configuration for commonroad and sumo simulation
        def load_reset_config(path):
            fr = CommonRoadFileReader(os.path.join(path,'cr_map.xml'))
            scenario, planning_problem_set = fr.open()
            problem_dict = {'scenario': scenario, 'planning_problem_set': planning_problem_set}
            return problem_dict

        self.all_problem_dict = load_reset_config(test_reset_config_path)

        print(f"Testing on {test_reset_config_path}")

        self.output_path = output_path

        self.termination = Termination(self.configs)
        self.terminated = False
        self.termination_reason = None

        # Ego vehicle
        self.ego_action, self.action_space = action_constructor(self.action_configs, self.vehicle_params)

        # Reward function
        self.reward_function: Reward = reward_constructor.make_reward(self.configs)

        # SUMO simulator
        self.sumo_map_path = test_reset_config_path
        self.sumo_cr_simulator_local = sumo_cr_simulator(self.configs, sumo_config)
        self.episode = 1
        self.worker_id = 0
        self.single_input_path = None

        print("Initialization done")

    @property
    def observation_space(self):
        return self.sumo_cr_simulator_local.observation_space()

    @property
    def observation_dict(self):
        return self.sumo_cr_simulator_local.observation_dict()

    def seed(self, seed=Union[None, int]):
        self.action_space.seed(seed)

    def set_episode(self, episode):
        self.episode = episode

    def set_worker_id(self, worker_id):
        self.worker_id = worker_id

    def set_single_input_path(self, single_input_path):
        self.single_input_path = single_input_path

    def reset(self) -> np.ndarray:
        """
        Reset the environment.

        :return: initial observation
        """

        # Set scenario and planning problem
        self._set_scenario_problem()
        self.step_ratio = int(self.planner_dt / self.sumo_cr_simulator_local.sumo_config.dt)
        self.ego_action.reset(self.planning_problem.initial_state, self.sumo_cr_simulator_local.sumo_config.dt)

        # reset sumo simulation
        interactive_scenario_path= os.path.join(self.sumo_map_path)
        initial_observation = self.sumo_cr_simulator_local.reset(interactive_scenario_path=interactive_scenario_path, 
                                                                 scenario=self.scenario, 
                                                                 planning_problem_set=self.planning_problem_set, 
                                                                 planning_problem=self.planning_problem, 
                                                                 ego_vehicle=self.ego_action.vehicle, 
                                                                 reset_config=self.reset_config, 
                                                                 output_file=self.output_path,
                                                                 episode=self.episode,
                                                                 worker_id=self.worker_id,
                                                                 single_input_path=self.single_input_path)

        # reset commonroad simulation
        self._set_initial_goal_reward()
        self.terminated = False
        self.reward_function.reset(self.observation_dict, self.ego_action)
        self.termination.reset(self.observation_dict, self.ego_action)
        self.v_ego_mean = self.ego_action.vehicle.state.velocity
        self.observation_list = [self.observation_dict]

        # reset result collection dict
        self.results = {'scenario': None, 'ego_vehicles': None, 'prediction_dicts': None, 'action_dict': None, 'scenario_without_occupancy': None}
        self.prediction_dicts = {}
        self.prediction_dict_all_steps = {}
        self.action_dict = []

        return initial_observation

    @property
    def current_step(self):
        # current time step from sumo simulation
        return self.sumo_cr_simulator_local.current_step()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Propagate to next time step, compute next observations, reward and status.
        """

        # update ego info
        if len(self.action_dict) > 0:
            self.action_dict[self.current_step].append([self.ego_action.vehicle.state.position, 
                                                        self.ego_action.vehicle.state.velocity, 
                                                        self.ego_action.vehicle.state.orientation])

        if self.action_configs['action_type'] == "continuous":
            action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        
        # Make action and observe result
        self.ego_action.step(action)
        self.sumo_cr_simulator_local.step(self.ego_action.vehicle)
        observation = self.sumo_cr_simulator_local.observation()

        # Check for termination
        done, reason, termination_info = self.termination.is_terminated(self.observation_dict, self.ego_action)
        if reason is not None:
            self.termination_reason = reason

        if done:
            self.terminated = True

        # Calculate reward
        reward = self.reward_function.calc_reward(self.observation_dict, self.ego_action)

        # Update info
        self.v_ego_mean += self.ego_action.vehicle.state.velocity
        self.observation_list.append(self.observation_dict)
        info = {
            "scenario_name": str(self.episode),
            "chosen_action": action,
            "current_episode_time_step": self.current_step,
            "max_episode_time_steps": self.sumo_cr_simulator_local.simulation.episode_length,
            "termination_reason": self.termination_reason,
            "v_ego_mean": self.v_ego_mean / self.current_step,
            "observation_list": self.observation_list
        }
        info.update(termination_info)

        if self.configs["surrounding_configs"]["observe_lane_circ_surrounding"] \
                or self.configs["surrounding_configs"]["observe_lane_rect_surrounding"]:
            info["ttc_follow"], info["ttc_lead"] = CommonroadEnv_SUMO.get_ttc_lead_follow(self.observation_dict)

        if termination_info["is_collision"] and self.check_collision_type:
            info = check_collision_type(info, self.ego_action.vehicle, 
                                        self.sumo_cr_simulator_local.simulation, 
                                        self.sumo_cr_simulator_local.simulation.current_scenario_full, 
                                        self.episode, 
                                        self.lane_change_time_threshold, 
                                        self.sumo_cr_simulator_local.simulation.local_ccosy)
            
        return observation, reward, done, info
    
    def collect_action(self, action, info):
        # collect the ego action and info for the video rendering
        for _ in range(self.step_ratio):
            self.action_dict.append([action, info])
    
    def collect_prediction_dict(self, prediction_dict):
        # get the prediction dict from SPOT
        self.prediction_dicts[self.current_step] = prediction_dict

    def transform_prediction_dict(self):
        # fill prediction dict for all time steps

        for time_step, obstacles in self.prediction_dicts.items():
            tmp_obstacles = {}
            for o in obstacles:
                tmp_obstacles[o.obstacle_id] = o
            for s in range(self.step_ratio):
                new_step = time_step + s
                new_obstacles = []
                for o in copy.deepcopy(obstacles):
                    occ_list = []
                    for time in range(self.step_ratio):
                        occ_list.append(tmp_obstacles[o.obstacle_id].occupancy_at_time(s + time + 1).shape)
                    occ_shape = ShapeGroup(occ_list)
                    o.prediction._occupancy_set = [Occupancy(new_step, occ_shape)]
                    new_obstacles.append(o)
                self.prediction_dict_all_steps[new_step] = new_obstacles

    def get_results(self, done=None, info=None):
        # get the results for video and analysis

        if hasattr(self.sumo_cr_simulator_local.simulation.sumo_sim, 'sim'):
            self.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.episode_info["end_time"] = self.sumo_cr_simulator_local.simulation.sumo_sim.sim.get_time()-self.sumo_cr_simulator_local.simulation.sumo_sim.sim.step_size

            # get termination reason
            reason, collision_ids = self.transform_info(info)
            self.sumo_cr_simulator_local.simulation.sumo_sim.sim.stop_reason = reason
            self.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.get_snapshot_info()
            self.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.get_terminate_info(done, reason, collision_ids)

        # get the results from sumo simulation
        scenario, ego_vehicles = self.sumo_cr_simulator_local.get_results()
        self.results['scenario_without_occupancy'] = scenario
        self.results['ego_vehicles'] = ego_vehicles

        # draw the occupancy set for each obstacle
        if len(self.prediction_dicts.keys()) > 0:
            self.transform_prediction_dict()
            obstacle_occupancy = {}
            for time_step, obstacles in self.prediction_dict_all_steps.items():
                for o in obstacles:
                    obstacle_occupancy[o.obstacle_id] = {}
            for time_step, obstacles in self.prediction_dict_all_steps.items():
                for o in obstacles:
                    obstacle_occupancy[o.obstacle_id][time_step] = o.occupancy_at_time(time_step)
            max_id = 30000 # assert that the id is not used
            for o in copy.deepcopy(scenario.dynamic_obstacles):
                id = o.obstacle_id
                o_new = DynamicObstacle(max_id, o.obstacle_type, o.obstacle_shape, o.initial_state, o.prediction)
                max_id += 1
                if id in obstacle_occupancy.keys():
                    if 0 in obstacle_occupancy[id].keys():
                        o_new._initial_occupancy_shape = obstacle_occupancy[id][0].shape
                    o_new.prediction._occupancy_set = []
                    for s in obstacle_occupancy[id].keys():
                        o_new.prediction._occupancy_set.append(obstacle_occupancy[id][s])
                scenario.add_objects(o_new)

        self.results['scenario'] = scenario
        self.results['prediction_dicts'] = self.prediction_dicts
        self.results['action_dict'] = self.action_dict

    def transform_info(self, info):
        # get the termination reason and collision ids from commonroad info
        additional_info = {"collision_id": []}
        if info['termination_reason'] == "is_off_road":
            reason = {6: "CAV leaves lane"}
        elif info['termination_reason'] == "is_collision":
            reason = {1: "CAV and BV collision"}
            ids = self.get_collision_id(self.ego_action.vehicle, self.sumo_cr_simulator_local.simulation)
            additional_info["collision_id"] = ids
        elif info['termination_reason'] == "is_timeout" or info['termination_reason'] == "is_friction_violation":
            reason = {7: "CAV too slow"}
        elif info['termination_reason'] == "is_goal_reached":
            reason = {4: "CAV reaches 800 m"}
        elif info['termination_reason'] == "infeasible":
            reason = {10: "CAV infeasible"}
        else:
            reason = {}
        return reason, additional_info

    def render_video(self, path=None, filename='test'):
        # render video for the simulation
        if path is None:
            path = self.output_path
        self.sumo_cr_simulator_local.create_video_for_simulation(self.results['scenario'], path, self.results['ego_vehicles'], filename, self.action_dict)

    def _set_scenario_problem(self) -> None:
        """
        Select scenario and planning problem.
        """
        problem_dict = self.all_problem_dict
        self.scenario = problem_dict['scenario']
        self.planning_problem_set = problem_dict["planning_problem_set"]
        self.planning_problem: PlanningProblem = random.choice(
            list(problem_dict["planning_problem_set"].planning_problem_dict.values())
        )
        self.reset_config = generate_reset_config(self.scenario, open_lane_ends=False)

        # Set name format for visualization
        self.file_name_format = str(self.episode) + "_ts_%03d"

    def _set_initial_goal_reward(self) -> None:
        """
        Set ego vehicle and initialize its status.
        """
        # Compute initial distance to goal for normalization if required
        if self.reward_type == "dense_reward":  # or "hybrid_reward":
            distance_goal_long, distance_goal_lat = self.sumo_cr_simulator_local.simulation.get_long_lat_distance_to_goal(
                self.ego_action.vehicle.state.position)
            self.initial_goal_dist = np.sqrt(distance_goal_long ** 2 + distance_goal_lat ** 2)

            # Prevent cases where the ego vehicle starts in the goal region
            if self.initial_goal_dist < 1.0:
                warnings.warn("Ego vehicle starts in the goal region")
                self.initial_goal_dist = 1.0

    @staticmethod
    def get_ttc_lead_follow(observation_dict):
        idx_follow = 1
        idx_lead = 4

        def get_ttc(p_rel, v_rel):
            if np.isclose(v_rel, 0.):
                return np.inf
            else:
                return p_rel / -v_rel

        # lane_based_v_rel = v_lead - v_follow
        # ttc: (s_lead - s_follow) / (v_follow - v_lead)
        ttc_follow = get_ttc(observation_dict["lane_based_p_rel"][idx_follow],
                             observation_dict["lane_based_v_rel"][idx_follow])
        ttc_lead = get_ttc(observation_dict["lane_based_p_rel"][idx_lead],
                           observation_dict["lane_based_v_rel"][idx_lead])

        return ttc_follow, ttc_lead
    
    @staticmethod
    def get_collision_id(ego_vehicle: Vehicle, observation_collector: simulate_scenario):
        # get the collision id from the commonroad scenario and collision checker

        collision_ego_vehicle = ego_vehicle.collision_object
        collision_objects = observation_collector.get_collision_checker().find_all_colliding_objects(collision_ego_vehicle)
        current_step = ego_vehicle.current_time_step

        obstacle_state_dict = dict()

        # for obstacle in observation_collector.current_scenario_full.obstacles:
        #    if obstacle.state_at_time(1):
        #        obstacle_start_step = obstacle.state_at_time(1).time_step - 1
        #    else:
        #        obstacle_start_step = 0
        #   obstacle_state = obstacle.state_at_time(current_step - obstacle_start_step)
        #    obstacle_state_pre = obstacle.state_at_time(current_step-1)
        #    if obstacle_state and obstacle_state_pre:
        #        obstacle_state.position = (obstacle_state.position + obstacle_state_pre.position) / 2
        #    if obstacle_state is not None:
        #        obstacle_state_dict[obstacle] = obstacle_state

        for obstacle in observation_collector.current_scenario.obstacles:
            obstacle_state = copy.deepcopy(obstacle.initial_state)
            obstacle_state.time_step = current_step
            obstacle_state_dict[obstacle.obstacle_id] = obstacle_state

        collision_obstacles = []
        for collision_object in collision_objects:
            collision_shape = collision_object.obstacle_at_time(ego_vehicle.current_time_step)
            if collision_shape.collide(observation_collector.road_edge["boundary_collision_object"]):
                continue
            if isinstance(collision_shape, pycrcc.RectOBB) or isinstance(collision_shape, pycrcc.Circle):
                center = collision_shape.center()
            elif isinstance(collision_shape, pycrcc.RectAABB):
                x_min, x_max = collision_shape.min_x(), collision_shape.max_x()
                y_min, y_max = collision_shape.min_y(), collision_shape.max_y()
                center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
            else:
                print('unknown type of collision shape')
            for obstacle_id, obstacle_state in obstacle_state_dict.items():
                if np.allclose(center, obstacle_state.position):
                    collision_obstacles.append(observation_collector.sumo_sim.ids_cr2sumo['obstacleVehicle'][obstacle_id])

        return collision_obstacles
