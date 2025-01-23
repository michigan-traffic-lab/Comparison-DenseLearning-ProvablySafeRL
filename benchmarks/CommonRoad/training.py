import os
import re
import sys
import yaml
import random
import glob
import numpy as np
os.environ["KMP_AFFINITY"] = "none"
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pickle
import argparse
import gym
import csv
import numpy as np

from src.auxiliary.ReachableSetManager import ReachableSetManager
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines.common.callbacks import CheckpointCallback

from commonroad_rl.utils_run.callbacks import SaveVecNormalizeCallback
from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv


class UnsafeWrapper(gym.Wrapper):
    """wrapper class for the f1tenth gym"""

    def __init__(self, env: ReplanningWrapper, reachable_set_manager: ReachableSetManager):
        """class constructor"""

        # call constructor of parent class
        super().__init__(env)

        # safe object properties
        self.reachable_set_manager = reachable_set_manager

    def step(self, alpha):
        assert np.all(np.abs(alpha)<=1.)

        u = self.reachable_set_manager.factor2control(alpha)

        return self.env.step(u)

class SafeWrapper(gym.Wrapper):
    """wrapper class for the f1tenth gym"""

    def __init__(self, env: ReplanningWrapper, reachable_set_manager: ReachableSetManager):
        """class constructor"""

        # call constructor of parent class
        super().__init__(env)

        # safe object properties
        self.reachable_set_manager = reachable_set_manager

    def step(self, alpha):
        assert np.all(np.abs(alpha)<=1.)

        u = self.reachable_set_manager.factor2control(alpha)

        return self.env.step(u)


def test_wrapper():
    # create basic commonroad-rl gym
    scenario_path = "/home/wangx/data/highD-dataset-v1.0/converted_extend_width_2/fixed_pickles" #"/home/xiao/projects/safe_RL/reachability_safe_rl/code/benchmarks/CommonRoad/scenarios/pickle"

    env = gym.make(
        "commonroad-v1",
        meta_scenario_path=os.path.join(scenario_path, "meta_scenario"),
        train_reset_config_path=os.path.join(scenario_path, "problem_tmp"),
        vehicle_params={"vehicle_model": 6}
    )

    planner = ReachableSetManager('commonRoad')
    env = ReplanningWrapper(env, planning_horizon=1., planner_dt=0.48)
    env = UnsafeWrapper(env, planner)

    print(env.action_space)

    initial_obs = env.reset()
    env.render()
    done = False
    for i in range(int(1e3)):
        alpha = env.action_space.sample()
        alpha = np.zeros(shape=alpha.shape)
        obs, step_reward, done, info = env.step(alpha)
        env.render()
        if done:
            break
            env.reset()


def train_model():

    planning_horizon = 0.8
    planning_dt = 0.4

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scenario_path", "-in", type=str,
        default="/home/xiao/projects/safe_RL/reachability_safe_rl/code/benchmarks/CommonRoad/scenarios/pickle",
        help="path to scenario pickle folder"
    )
    parser.add_argument("--log_dir", "-log", type=str, default="./log", help="path to store training results")
    parser.add_argument("--n_envs", "-n", type=int, default=1, help="number of parallel processes for training")
    parser.add_argument("--save_freq", type=int, default=20000, help="number of steps for storing the model")
    parser.add_argument("--total_timesteps", "-step", type=int, default=int(1e6), help="number of total training steps")

    args = parser.parse_args()

    # create basic commonroad-rl gym
    scenario_path = args.scenario_path
    log_dir = args.log_dir
    n_envs = args.n_envs
    save_freq = args.save_freq
    total_timesteps = args.total_timesteps

    def make_env(rank, log_dir):

        def _init():
            env = gym.make(
                "commonroad-v1",
                meta_scenario_path=os.path.join(scenario_path, "meta_scenario"),
                train_reset_config_path=os.path.join(scenario_path, "problem", str(rank)),
                vehicle_params={"vehicle_model": 6}
            )

            env = ReplanningWrapper(env, planning_horizon=planning_horizon, planner_dt=planning_dt)
            env = UnsafeWrapper(env, reachable_set_manager=ReachableSetManager('commonRoad'))

            # Monitor wrapper to log learning curves
            log_file = os.path.join(log_dir, str(rank)) if log_dir is not None else None
            env = Monitor(env, log_file, info_keywords=("is_goal_reached", "is_collision", "is_time_out", "is_off_road"))

            return env

        return _init

    os.makedirs(log_dir, exist_ok=True)
    vec_envs = SubprocVecEnv([make_env(rank, log_dir) for rank in range(n_envs)], start_method="spawn")
    env = VecNormalize(vec_envs)

    callbacks = []

    # Account for the number of parallel environments
    callbacks.append(CheckpointCallback(save_freq=save_freq, save_path=log_dir, name_prefix="rl_model", verbose=True))
    callbacks.append(SaveVecNormalizeCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix="vecnormalize",
        verbose=True,
    ))

    model = PPO2(policy='MlpPolicy', env=env)
    model.learn(total_timesteps=total_timesteps, log_interval=100000, callback=callbacks)
    model.save(log_dir + "/model")
    model.get_vec_normalize_env().save(os.path.join(log_dir, "vecnormalize.pkl"))


if __name__=='__main__':
    """main entry point"""
#    test_wrapper()
    train_model()
