import os
import torch
os.environ["KMP_AFFINITY"] = "none"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import gym
import glob
import random
import pickle
from PIL import Image
import imageio.v2 as imageio
import argparse
import numpy as np
from agent.UnsafeReinforcementPlanner import UnsafeReinforcementPlanner
from auxiliary.utils import ReplanningWrapper

from src.SafeReinforcementPlanner import SafeReinforcementPlanner
from src.auxiliary.safety import PlanningInfeasibleError
from src.auxiliary.utils import MultiLayerPerceptron
from commonroad.scenario.trajectory import State

import gym_commonroad_sumo # do not remove this line
from gym_commonroad_sumo.commonroad_sumo_env import CommonroadEnv_SUMO
from sumo_config.generate_cr_xml_from_sumo import generate_cr_map
from sumo_config.default_sumo_config import DefaultConfig

import warnings
warnings.filterwarnings("ignore")

class ObservationNormalizer:
    def __init__(self, path_normalize):
        norm = {}
        with open(path_normalize, 'rb') as f:
            norm['mean'], norm['var'], norm['epsilon'], norm['clip'] = pickle.load(f) # mean(35,), var(35,), epsilon: 1e-8, clip: 10.0
            self.vecnorm = norm

    def normalize_obs(self, obs):
        return np.clip(
            (obs - self.vecnorm['mean']) / np.sqrt(self.vecnorm['var'] + self.vecnorm['epsilon']),
            -self.vecnorm['clip'],
            self.vecnorm['clip'])


def make_gif(path: str, benchmark_id,
             file_save_name="animation", duration: float = 0.1):

    images = []
    filenames = sorted(glob.glob(os.path.join(path, "*.png")))

    for filename in filenames:
        image = Image.open(filename)
        if len(images) == 0:
            shape = image.size
        elif not image.size == shape:
            image = image.resize(shape)
        images.append(image)

    imageio.mimsave(os.path.join(path, "../", file_save_name + f"_{benchmark_id}.gif"), images, duration=duration)

def run_trained_agent():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--load_path", "-model", type=str, default="./agent")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--source_path", type=str, default="./sumo_config/sumo_maps/3LaneHighway")
    parser.add_argument("--mode", type=str, default="safe")
    args = parser.parse_args()

    MODE = args.mode

    planning_horizon = 0.8
    planner_dt = 0.4

    # create basic commonroad-rl gym
    source_path = args.source_path
    generate_cr_map(source_path)

    os.makedirs(args.output_path, exist_ok=True)

    def env_fn():
        env = gym.make(
            'commonroad_sumo-v0',
            test_reset_config_path=source_path,
            vehicle_params={"vehicle_model": 6},
            config_file=os.path.join(args.load_path, "environment_configurations.yml"),
            output_path=args.output_path,
            sumo_config=DefaultConfig(),
        )
        env = ReplanningWrapper(env, planning_horizon=planning_horizon, planner_dt=planner_dt)

        return env

    # pytorch dependencies
    env = env_fn()

    # load trained reinforcement learning agent
    model_path = os.path.join(args.load_path, "PyTorchModel")
    model = MultiLayerPerceptron(35, 4)
    model.load_state_dict(torch.load(model_path))
    model = model.eval()
    commonroad_env: CommonroadEnv_SUMO = env.env.env
    observation_normalizer = ObservationNormalizer(os.path.join(args.load_path, "normalization.pkl"))

    # initialize the planners
    unsafe_planner = UnsafeReinforcementPlanner("commonRoad3", model=model)
    # commonRoad: original reachable set; 
    # commonRoad2: regenerate by correct ego length = 5 and width = 1.8; 
    # commonRoad3: regenerate by new length and width, and adjust the control commands interval to [-4, 2] and [-0.2, 0.2]
    safe_planner = SafeReinforcementPlanner(unsafe_planner)

    random.seed(2)
    obs = env.reset()
    next_termination = True

    while True:

        safe_planner.reset(scenario=commonroad_env.sumo_cr_simulator_local.simulation._scenario,
        planning_problem_set=commonroad_env.planning_problem_set)
        safe_planner.set_prediction_steps(prediction_steps=int(planning_horizon/commonroad_env.sumo_cr_simulator_local.simulation._scenario.dt))

        done = False
        while not done:
            ego_state: State = commonroad_env.ego_action.vehicle.state
            observation_dict = {
                'xPosition': ego_state.position[0],
                'yPosition': ego_state.position[1],
                'velocity': ego_state.velocity,
                'orientation': ego_state.orientation,
                'time': ego_state.time_step * commonroad_env.sumo_cr_simulator_local.simulation._scenario.dt,
                'normalized_observation_vector': observation_normalizer.normalize_obs(obs)
            }

            if MODE == "unsafe":
                # unsafe planner
                alpha = unsafe_planner.plan(observation_dict)
                u = unsafe_planner.reachable_set_manager.factor2control(alpha)[0]
            elif MODE == "safe":
                # safe planner
                np.random.seed(1)
                random.seed(1)
                try:
                    u, intended_input, _1, _2 = safe_planner.plan(observation_dict, 
                            scenario=commonroad_env.sumo_cr_simulator_local.simulation.current_scenario) # control inputs
                except PlanningInfeasibleError:
                    env.reset()
                    info[0]["termination_reason"] = "infeasible"
                    print(info[0]["termination_reason"])
                    break

            obs, reward, done, info = env.step([[u]])
        
        env.get_results()
        env.render_video()
        obs = env.reset()

        if next_termination:
            break


if __name__ == "__main__":
    run_trained_agent()
