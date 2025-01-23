import os, shutil
import torch
import time
os.environ["KMP_AFFINITY"] = "none"
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import gym
import random
import pickle
import argparse
import numpy as np
from agent.UnsafeReinforcementPlanner import UnsafeReinforcementPlanner
from agent.UnsafePlanner_observor import Unsafeplanner_observor
from auxiliary.utils import ReplanningWrapper

from src.SafeReinforcementPlanner import SafeReinforcementPlanner
from src.auxiliary.safety import PlanningInfeasibleError
from src.auxiliary.utils import MultiLayerPerceptron
from commonroad.scenario.trajectory import State

import gym_commonroad_sumo  # do not remove this line
from gym_commonroad_sumo.commonroad_sumo_env import CommonroadEnv_SUMO
from sumo_config.generate_cr_xml_from_sumo import generate_cr_map
from sumo_config.default_sumo_config import DefaultConfig
from gym_commonroad_sumo.simulation.transfer_replay_scenario import transfer_all_replay_scenario

import warnings
warnings.filterwarnings("ignore")

class ObservationNormalizer:
    def __init__(self, path_normalize):
        norm = {}
        with open(path_normalize, "rb") as f:
            norm["mean"], norm["var"], norm["epsilon"], norm["clip"] = pickle.load(
                f
            )  # mean(35,), var(35,), epsilon: 1e-8, clip: 10.0
            self.vecnorm = norm

    def normalize_obs(self, obs):
        return np.clip(
            (obs - self.vecnorm["mean"])
            / np.sqrt(self.vecnorm["var"] + self.vecnorm["epsilon"]),
            -self.vecnorm["clip"],
            self.vecnorm["clip"],
        )

def run_trained_agent():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--load_path", "-model", type=str, default="./agent")
    parser.add_argument("--output_path", type=str, default="./results3")
    parser.add_argument("--source_path", type=str, default="./sumo_config/sumo_maps/3LaneHighway")
    NDE_path = os.path.abspath('source_data/NDE_Data')
    parser.add_argument("--NDD_DATA_PATH", type=str, default=NDE_path, help="Absolute path to the NDD data")
    parser.add_argument("--episode_num", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--replay_scenario_root_path", type=str, default='results_all/results_safeagent_without_infeasible_ending')
    # All the saved scenarios in the replay_scenario_root_path will be replayed. Each scenario should contains all the info saved automatically.
    parser.add_argument("--safedriver_only", action="store_true") # if True, only safedriver will be tested; if False, both safe planner and safedriver will be tested
    parser.add_argument("--base", type=str, default="safe planner") # the compared model used in the original scenario
    args = parser.parse_args()

    SafeDriverONLY = args.safedriver_only
    Base = args.base

    os.environ['NDD_DATA_PATH'] = args.NDD_DATA_PATH

    planning_horizon = 0.8
    planner_dt = 0.4
    safe_driver_dt = 0.1

    # create basic commonroad source files
    source_path = args.source_path
    generate_cr_map(source_path)

    scenario_replay_path = os.path.join(args.output_path,'scenario_replay')
    if not os.path.isdir(scenario_replay_path) or len(os.listdir(scenario_replay_path)) == 0:
        transfer_all_replay_scenario(args.replay_scenario_root_path, scenario_replay_path)

    if os.path.isdir(os.path.join(args.output_path, 'saved_data')):
        shutil.rmtree(os.path.join(args.output_path, 'saved_data'))
    if os.path.isdir(os.path.join(args.output_path, 'crash')):
        shutil.rmtree(os.path.join(args.output_path, 'crash'))
    if os.path.isdir(os.path.join(args.output_path, 'tested_and_safe')):
        shutil.rmtree(os.path.join(args.output_path, 'tested_and_safe'))

    def env_fn():
        env_safe_planner = gym.make(
            "commonroad_nade-v0",
            test_reset_config_path=source_path,
            vehicle_params={"vehicle_model": 6},
            config_file=os.path.join(args.load_path, "environment_configurations.yml"),
            output_path=args.output_path,
            sumo_config=DefaultConfig(),
            planner_dt=planner_dt,
        )
        env_safe_planner = ReplanningWrapper(
            env_safe_planner, planning_horizon=planning_horizon, planner_dt=planner_dt
        )
        env_safedriver = gym.make(
            "commonroad_nade-v0",
            test_reset_config_path=source_path,
            vehicle_params={"vehicle_model": 6},
            config_file=os.path.join(args.load_path, "environment_configurations.yml"),
            output_path=args.output_path,
            sumo_config=DefaultConfig(),
            planner_dt=safe_driver_dt,
        )
        env_safedriver = ReplanningWrapper(
            env_safedriver, planning_horizon=safe_driver_dt, planner_dt=safe_driver_dt
        )

        return env_safe_planner, env_safedriver

    env_safe_planner, env_safedriver = env_fn()

    # load trained reinforcement learning agent
    model_path = os.path.join(args.load_path, "PyTorchModel")
    model = MultiLayerPerceptron(35, 4)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # initialize the environment
    commonroad_env_safe_planner: CommonroadEnv_SUMO = env_safe_planner.env.env
    commonroad_env_safedriver: CommonroadEnv_SUMO = env_safedriver.env.env
    observation_normalizer = ObservationNormalizer(
        os.path.join(args.load_path, "normalization.pkl")
    )

    # initialize the planners
    unsafe_planner = UnsafeReinforcementPlanner("commonRoad3", model=model)
    # commonRoad: original reachable set; 
    # commonRoad2: regenerate by correct ego length = 5 and width = 1.8; 
    # commonRoad3: regenerate by new length and width, and adjust the control commands interval to [-4, 2] and [-0.2, 0.2]
    if not SafeDriverONLY:
        safe_planner = SafeReinforcementPlanner(unsafe_planner)

    from safedriver.SafeDriver import SafeDriver
    safe_driver = SafeDriver(unsafe_planner, base='unsafe')
    safe_driver.set_time_step(safe_driver_dt, unsafe_planner_dt=planner_dt)

    for folder in os.listdir(scenario_replay_path):
        episode = int(folder.split('_')[-1])
        single_input_path = os.path.join(scenario_replay_path, folder)
        # initialize the episode
        print(f"Episode {episode}")
        seed = args.episode_num * args.worker_id + episode
        random.seed(seed)
        np.random.seed(seed)
        data_save_path = os.path.join(args.output_path, 'saved_data', 'worker_'+str(args.worker_id), 'episode_'+str(episode))
        os.makedirs(data_save_path, exist_ok=True)
        
        # give new episode id and worker id to the environment
        commonroad_env_safedriver.set_episode(episode)
        commonroad_env_safedriver.set_worker_id(args.worker_id)
        commonroad_env_safedriver.set_single_input_path(single_input_path)
        if not SafeDriverONLY:
            commonroad_env_safe_planner.set_episode(episode)
            commonroad_env_safe_planner.set_worker_id(args.worker_id)
            commonroad_env_safe_planner.set_single_input_path(single_input_path)

        # safe driver
        obs_safedriver = env_safedriver.reset()
        safe_driver.reset()
        done = False
        while not done:
            ego_state_safedriver: State = commonroad_env_safedriver.ego_action.vehicle.state
            observation_dict_safedriver = {
                "xPosition": ego_state_safedriver.position[0],
                "yPosition": ego_state_safedriver.position[1],
                "velocity": ego_state_safedriver.velocity,
                "orientation": ego_state_safedriver.orientation, # east is 0, counter-clockwise is positive, yaw rate: counter-clockwise is positive
                "time": ego_state_safedriver.time_step
                * commonroad_env_safedriver.sumo_cr_simulator_local.simulation._scenario.dt,
                "normalized_observation_vector": observation_normalizer.normalize_obs(
                    obs_safedriver
                ),
                'obs': obs_safedriver.tolist(),
            }

            # safe driver
            if observation_dict_safedriver["time"] == 0: # env.get_observation is not available at the first step
                drl_obs = None
                neuralmetric_obs = None
                current_env = None
            else:
                drl_obs = commonroad_env_safedriver.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_drl()
                neuralmetric_obs = commonroad_env_safedriver.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_neuralmetric()
                current_env = commonroad_env_safedriver.sumo_cr_simulator_local.simulation.sumo_sim.sim.env
            
            scenario = commonroad_env_safedriver.sumo_cr_simulator_local.simulation.current_scenario
            u_safedriver, _1, _2 = safe_driver.plan(
                observation_dict_safedriver, ego_state_safedriver, drl_obs, neuralmetric_obs, current_env, scenario
            )  # control inputs
            u = u_safedriver
            if safe_driver.unsafe_control_implemented_step == 0:
                replay_info = 'controlled by safedriver'
            else:
                replay_info = 'controlled by unsafe planner'
            replay_info += f', comparison model: {Base} \n '
            commonroad_env_safedriver.collect_action(u, replay_info + safe_driver.info)

            obs_safedriver, reward, done, info_safedriver = env_safedriver.step([[u]])

        env_safedriver.get_results(done, info_safedriver)

        # safe planner
        if not SafeDriverONLY:
            obs_safe_planner = env_safe_planner.reset()
            safe_planner.reset(scenario=commonroad_env_safe_planner.sumo_cr_simulator_local.simulation._scenario,
                                planning_problem_set=commonroad_env_safe_planner.planning_problem_set, 
                                path=data_save_path,
                                draw_flag=False)
            safe_planner.set_prediction_steps(
                prediction_steps=int(
                    planning_horizon
                    / commonroad_env_safe_planner.sumo_cr_simulator_local.simulation._scenario.dt
                )
            )
            done = False
            while not done:
                ego_state_safe_planner: State = commonroad_env_safe_planner.ego_action.vehicle.state
                observation_dict_safe_planner = {
                    "xPosition": ego_state_safe_planner.position[0],
                    "yPosition": ego_state_safe_planner.position[1],
                    "velocity": ego_state_safe_planner.velocity,
                    "orientation": ego_state_safe_planner.orientation, # east is 0, counter-clockwise is positive, yaw rate: counter-clockwise is positive
                    "time": ego_state_safe_planner.time_step
                    * commonroad_env_safe_planner.sumo_cr_simulator_local.simulation._scenario.dt,
                    "normalized_observation_vector": observation_normalizer.normalize_obs(
                        obs_safe_planner
                    ),
                    'obs': obs_safe_planner.tolist(),
                }

                # unsafe planner
                alpha = unsafe_planner.plan(observation_dict_safe_planner)
                u = unsafe_planner.reachable_set_manager.factor2control(alpha)[0]
                replay_info = 'controlled by unsafe planner'
                # safe planner
                u_safe_planner, intended_input, _1, _2 = safe_planner.plan(
                    observation_dict_safe_planner,
                    scenario=commonroad_env_safe_planner.sumo_cr_simulator_local.simulation.current_scenario,
                )  # control inputs
                #commonroad_env_safe_planner.collect_prediction_dict(safe_planner.prediction_dict)
                if (u != u_safe_planner).any():
                    u = u_safe_planner
                    replay_info = 'controlled by safe planner'
                replay_info += f', comparison model: {Base} \n '
                commonroad_env_safe_planner.collect_action(u, replay_info + safe_planner.info)

                obs_safe_planner, reward, done, info_safe_planner = env_safe_planner.step([[u]])
            
            env_safe_planner.get_results(done, info_safe_planner)

        # save the simulation info for critical scenarios
        env_safedriver.render_video(path=os.path.join(commonroad_env_safedriver.output_path, 'saved_data', 'videos'), filename=f"episode_{episode}_safedriver_collision_{info_safedriver['is_collision']}_time_{int(time.time())}")
        if not SafeDriverONLY:
            env_safe_planner.render_video(path=os.path.join(commonroad_env_safe_planner.output_path, 'saved_data', 'videos'), filename=f"episode_{episode}_safe_planner_collision_{info_safe_planner['is_collision']}_time_{int(time.time())}")

        with open(os.path.join(data_save_path, 'safedriver_simulation_info.pkl'), 'wb') as file:
            pickle.dump(commonroad_env_safedriver.results, file)
        with open(os.path.join(data_save_path, 'safedriver_planner_info.pkl'), 'wb') as file:
            pickle.dump(safe_driver.save_info, file)
        if not SafeDriverONLY:
            with open(os.path.join(data_save_path, 'safe_planner_simulation_info.pkl'), 'wb') as file:
                pickle.dump(commonroad_env_safe_planner.results, file)
            with open(os.path.join(data_save_path, 'safe_planner_planner_info.pkl'), 'wb') as file:
                pickle.dump(safe_planner.save_info, file)

if __name__ == "__main__":
    run_trained_agent()
