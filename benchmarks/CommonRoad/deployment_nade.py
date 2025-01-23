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
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--source_path", type=str, default="./sumo_config/sumo_maps/3LaneHighway")
    NDE_path = os.path.abspath('source_data/NDE_Data')
    parser.add_argument("--NDD_DATA_PATH", type=str, default=NDE_path, help="Absolute path to the NDD data")
    parser.add_argument("--episode_num", type=int, default=500)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--mode", type=str, default="safedriver") # unsafe, safe, safedriver, finetune
    parser.add_argument("--base", type=str, default="unsafe") # the base planner of safedriver, 'unsafe' or 'safe'
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    MODE = args.mode
    TEST = args.test
    BASE = args.base
    
    #if TEST:
    #    args.output_path = "./test"

    os.environ['NDD_DATA_PATH'] = args.NDD_DATA_PATH

    planning_horizon = 0.8
    planner_dt = 0.4
    safe_driver_dt = 0.1

    # create basic commonroad source files
    source_path = args.source_path
    generate_cr_map(source_path)

    def env_fn():
        if MODE == "unsafe" or MODE == "safe":
            env = gym.make(
                "commonroad_nade-v0",
                test_reset_config_path=source_path,
                vehicle_params={"vehicle_model": 6},
                config_file=os.path.join(args.load_path, "environment_configurations.yml"),
                output_path=args.output_path,
                sumo_config=DefaultConfig(),
                planner_dt=planner_dt,
            )
            env = ReplanningWrapper(
                env, planning_horizon=planning_horizon, planner_dt=planner_dt
            )
        elif MODE == "safedriver" or MODE == "finetune":
            env = gym.make(
                "commonroad_nade-v0",
                test_reset_config_path=source_path,
                vehicle_params={"vehicle_model": 6},
                config_file=os.path.join(args.load_path, "environment_configurations.yml"),
                output_path=args.output_path,
                sumo_config=DefaultConfig(),
                planner_dt=safe_driver_dt,
            )
            env = ReplanningWrapper(
                env, planning_horizon=safe_driver_dt, planner_dt=safe_driver_dt
            )

        return env

    # pytorch dependencies
    env = env_fn()

    # load trained reinforcement learning agent
    model_path = os.path.join(args.load_path, "PyTorchModel")
    model = MultiLayerPerceptron(35, 4)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    # initialize the environment
    commonroad_env: CommonroadEnv_SUMO = env.env.env
    observation_normalizer = ObservationNormalizer(
        os.path.join(args.load_path, "normalization.pkl")
    )

    # initialize the planners
    unsafe_planner_observor = Unsafeplanner_observor()
    unsafe_planner = UnsafeReinforcementPlanner("commonRoad3", model=model)
    # commonRoad: original reachable set; 
    # commonRoad2: regenerate by correct ego length = 5 and width = 1.8; 
    # commonRoad3: regenerate by new length and width, and adjust the control commands interval to [-4, 2] and [-0.2, 0.2]
    if MODE == 'safe':
        safe_planner = SafeReinforcementPlanner(unsafe_planner)
        end_with_infeasible = False
    else:
        safe_planner = None

    from safedriver.SafeDriver import SafeDriver
    if MODE == 'safedriver':
        if BASE == 'safe':
            safe_planner = SafeReinforcementPlanner(unsafe_planner)
            safe_driver = SafeDriver(safe_planner, base='safe')
            safe_driver.set_time_step(safe_driver_dt, unsafe_planner_dt=planner_dt)
        else:
            safe_driver = SafeDriver(unsafe_planner, base='unsafe')
            safe_driver.set_time_step(safe_driver_dt, unsafe_planner_dt=planner_dt)
    else:
        safe_driver = None
    
    if MODE == 'finetune':
        safe_driver = SafeDriver(unsafe_planner, base='unsafe')
        safe_driver.set_time_step(safe_driver_dt, unsafe_planner_dt=planner_dt)

    for episode in range(args.worker_id*20000, args.worker_id*20000 + args.episode_num):

        # initialize the episode
        print(f"Episode {episode}")
        seed = args.episode_num * args.worker_id + episode
        random.seed(seed)
        np.random.seed(seed)

        data_save_path = os.path.join(args.output_path, 'saved_data', 'worker_'+str(args.worker_id), 'episode_'+str(episode))
        os.makedirs(data_save_path, exist_ok=True)
        
        # give new episode id and worker id to the environment
        commonroad_env.set_episode(episode)
        commonroad_env.set_worker_id(args.worker_id)

        # reset the environment
        obs = env.reset()

        # reset the safe planner
        if MODE == 'safe':
            safe_planner.reset(scenario=commonroad_env.sumo_cr_simulator_local.simulation._scenario,
                                planning_problem_set=commonroad_env.planning_problem_set, 
                                path=data_save_path,
                                draw_flag=False)
            safe_planner.set_prediction_steps(
                prediction_steps=int(
                    planning_horizon
                    / commonroad_env.sumo_cr_simulator_local.simulation._scenario.dt
                )
            )
        elif MODE == 'safedriver':
            safe_driver.reset()
            if BASE == 'safe':
                safe_driver.planner.reset(scenario=commonroad_env.sumo_cr_simulator_local.simulation._scenario,
                                planning_problem_set=commonroad_env.planning_problem_set, 
                                path=data_save_path,
                                draw_flag=False)
                safe_driver.planner.set_prediction_steps(
                    prediction_steps=int(
                        planning_horizon
                        / commonroad_env.sumo_cr_simulator_local.simulation._scenario.dt
                    )
                )

        elif MODE == 'finetune':
            safe_driver.reset()
        
        done = False
        while not done:
            ego_state: State = commonroad_env.ego_action.vehicle.state
            observation_dict = {
                "xPosition": ego_state.position[0],
                "yPosition": ego_state.position[1],
                "velocity": ego_state.velocity,
                "orientation": ego_state.orientation, # east is 0, counter-clockwise is positive, yaw rate: counter-clockwise is positive
                "time": ego_state.time_step
                * commonroad_env.sumo_cr_simulator_local.simulation._scenario.dt,
                "normalized_observation_vector": observation_normalizer.normalize_obs(
                    obs
                ),
                'obs': obs.tolist(),
            }

            if MODE == "unsafe":
                # unsafe planner
                alpha = unsafe_planner.plan(observation_dict)
                u = unsafe_planner.reachable_set_manager.factor2control(alpha)[0]
            elif MODE == "safe":
                # safe planner
                u, intended_input, _1, _2 = safe_planner.plan(
                    observation_dict,
                    scenario=commonroad_env.sumo_cr_simulator_local.simulation.current_scenario,
                )  # control inputs
                commonroad_env.collect_prediction_dict(safe_planner.prediction_dict)
                commonroad_env.collect_action(u, safe_planner.info)
                if safe_planner.info == 'infeasible' and end_with_infeasible:
                    info["termination_reason"] = "infeasible"
                    done = True
                    print(info["termination_reason"])
                    break
            elif MODE == "safedriver":
                # safe driver
                if observation_dict["time"] == 0: # env.get_observation is not available at the first step
                    drl_obs = None
                    neuralmetric_obs = None
                    current_env = None
                else:
                    drl_obs = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_drl()
                    neuralmetric_obs = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_neuralmetric()
                    current_env = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env
                
                scenario = commonroad_env.sumo_cr_simulator_local.simulation.current_scenario
                u, _1, _2 = safe_driver.plan(
                    observation_dict, ego_state, drl_obs, neuralmetric_obs, current_env, scenario
                )  # control inputs
                if BASE == 'safe':
                    commonroad_env.collect_action(u, safe_driver.planner.info + '\n ' + safe_driver.info)
                else:
                    commonroad_env.collect_action(u, safe_driver.info)

            elif MODE == "finetune":
                alpha = unsafe_planner.plan(observation_dict)
                u = unsafe_planner.reachable_set_manager.factor2control(alpha)[0]
                
                if observation_dict["time"] == 0:
                    drl_obs = None
                    neuralmetric_obs = None
                    current_env = None
                else:
                    drl_obs = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_drl()
                    neuralmetric_obs = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.get_observation_neuralmetric()
                    current_env = commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env
                scenario = commonroad_env.sumo_cr_simulator_local.simulation.current_scenario
                safe_driver.plan(
                    observation_dict, ego_state, drl_obs, neuralmetric_obs, current_env, scenario
                )

            obs, reward, done, info = env.step([[u]])

        # get all the history info and stop the sumo simulation
        if MODE == 'safedriver' or MODE == 'finetune':
            for i in range(len(safe_driver.save_info['criticality'])):
                try:
                    commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.cav_info[round(i*safe_driver_dt,4)]['criticality'] = safe_driver.save_info['criticality'][i]
                    commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.cav_info[round(i*safe_driver_dt,4)]['NADE_critical'] = safe_driver.save_info['criticality'][i] >= safe_driver.criticality_thresh
                    commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.cav_info[round(i*safe_driver_dt,4)]['slightly_critical'] = safe_driver.save_info['criticality'][i] > 0.0

                    if safe_driver.save_info['criticality'][i] < safe_driver.criticality_thresh:
                        commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.cav_info[round(i*safe_driver_dt,4)]["CAV_action"] = {
                            "acceleration": safe_driver.save_info['safe_action'][i][0], 
                            "steering_angle": safe_driver.save_info['safe_action'][i][1], 
                            "additional_info": {
                                "rl_obs": safe_driver.save_info['drl_obs'][i], 
                                "veh_states": safe_driver.save_info['ego_state_list_sumo'][i], 
                                "RSS_states": None, 
                                "NN_metric_obs": safe_driver.save_info['neuralmetric_obs'][i],
                                "unsafe_planner_obs": safe_driver.save_info['observation'][i]['obs'],
                                "final_action": [safe_driver.save_info['u'][i][0], safe_driver.save_info['u'][i][1]],
                            }
                        }
                    else:
                        commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.info_extractor.cav_info[round(i*safe_driver_dt,4)]["CAV_action"] = {
                            "acceleration": safe_driver.save_info['safe_action'][i][0], 
                            "steering_angle": safe_driver.save_info['safe_action'][i][1], 
                            "additional_info": {
                                "rl_obs": safe_driver.save_info['drl_obs'][i], 
                                "veh_states": safe_driver.save_info['ego_state_list_sumo'][i], 
                                "RSS_states": None, 
                                "NN_metric_obs": safe_driver.save_info['neuralmetric_obs'][i],
                                "unsafe_planner_obs": safe_driver.save_info['observation'][i]['obs'],
                                "final_action": [safe_driver.save_info['u'][i][0], safe_driver.save_info['u'][i][1]],
                            }, 
                        }

                except:
                    pass

        env.get_results(done, info)

        # save the simulation info for critical scenarios
        infeasible = int(info["termination_reason"] == "infeasible")
        is_off_road = int(info["termination_reason"] == "is_off_road")
        if info['is_collision'] or infeasible or is_off_road or TEST:

            env.render_video(path=os.path.join(commonroad_env.output_path, 'saved_data', 'videos'), filename=f"worker_{args.worker_id}_episode_{episode}_collision_{info['is_collision']}_infeasible_{infeasible}_off_road_{is_off_road}_time_{int(time.time())}")

            with open(os.path.join(data_save_path, 'simulation_info.pkl'), 'wb') as file:
                pickle.dump(commonroad_env.results, file)

            if MODE == 'safe':
                with open(os.path.join(data_save_path, 'planner_info.pkl'), 'wb') as file:
                    pickle.dump(safe_planner.save_info, file)
            elif MODE == 'safedriver':
                with open(os.path.join(data_save_path, 'planner_info.pkl'), 'wb') as file:
                    pickle.dump(safe_driver.save_info, file)

        # delete the saved data if unnecessary
        else:
            shutil.rmtree(data_save_path)

if __name__ == "__main__":
    run_trained_agent()
