import os
import pathlib
from numpy import ndarray
import yaml

import gym_commonroad_sumo  # do not remove this line
from gym_commonroad_sumo.action import action_constructor
from gym_commonroad_sumo.commonroad_sumo_env import CommonroadEnv_SUMO
from gym_commonroad_sumo.file_reader import CommonRoadFileReader
from gym_commonroad_sumo.reward import reward_constructor
from gym_commonroad_sumo.reward.reward import Reward
from gym_commonroad_sumo.reward.termination import Termination

from gym_commonroad_sumo.simulation.nade_cr_simulation import nade_cr_simulator

class CommonRoadEnv_Nade(CommonroadEnv_SUMO):
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
        #print("Initialization started")

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

        def load_reset_config(path):
            fr = CommonRoadFileReader(os.path.join(path, "cr_map.xml"))
            scenario, planning_problem_set = fr.open()
            problem_dict = {
                "scenario": scenario,
                "planning_problem_set": planning_problem_set,
            }
            return problem_dict

        self.all_problem_dict = load_reset_config(test_reset_config_path)

        #print(f"Testing on {test_reset_config_path}")

        self.output_path = output_path
        os.makedirs(os.path.join(self.output_path,'crash'), exist_ok=True)

        self.termination = Termination(self.configs)
        self.terminated = False
        self.termination_reason = None

        self.ego_action, self.action_space = action_constructor(
            self.action_configs, self.vehicle_params
        )

        # Reward function
        self.reward_function: Reward = reward_constructor.make_reward(self.configs)

        # SUMO config
        self.sumo_map_path = test_reset_config_path
        self.sumo_cr_simulator_local = nade_cr_simulator(self.configs, sumo_config)
        self.episode = 1
        self.worker_id = 0
        self.single_input_path = None
        
        #print("Initialization done")

