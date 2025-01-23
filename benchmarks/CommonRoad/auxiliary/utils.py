import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import pickle
import numpy as np
import torch as th


from commonroad_rl.gym_commonroad.commonroad_env import CommonroadEnv
try:
    from stable_baselines.ppo2.ppo2 import PPO2
    from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
except:
    pass

from src.auxiliary.utils import MultiLayerPerceptron, get_model_and_vecnormalize_path


class ReplanningWrapper(gym.Wrapper):
    def __init__(self, env: CommonroadEnv, planning_horizon, planner_dt):
        """class constructor"""

        # call constructor of parent class
        super().__init__(env)
        self.planning_horizon = planning_horizon  # final time for simulation # TODO: time-out?
        self.planning_dt = planner_dt
        self.action_space = self.build_action_space()

    def build_action_space(self):
        N = int(self.planning_horizon // self.planning_dt)
        return gym.spaces.Box(
            low=np.tile([-1., -1.], (N)),
            high=np.tile([1., 1.], (N)),
            dtype=np.float32)

    def _denormalize(self, action: np.ndarray) -> np.ndarray:
        """
        Scales action from unit value to its true value.

        :param action: unit scaled value action
        :return: true value action
        """
        assert np.all(action <= 1. + 1e-6) and np.all(action >= -1. - 1e-6), f"action={action}"
        return self.env.ego_action.rescale_action(action)  # action * self.rescale_factor + self.rescale_bias

    def _normalize(self, u: np.ndarray) -> np.ndarray:
        """
        Scales action to unit value range.

        :param u: action to be scaled
        :return: scaled action
        """

        return (u - self.env.ego_action._rescale_bias) / self.env.ego_action._rescale_factor

    def step(self, u: np.array):
        """
        u: normalized control inputs of point mass model: a_x, a_y
        """
        # simulate the environment
        cnt = 0
        reward = 0.
        time = 0.
        planning_horizon = self.planning_dt

        while round(time, 2) < planning_horizon: # training with only executing the first input
            # change the control i nput
            if time >= (cnt + 1) * self.planning_dt:
                cnt += 1

            # simulate environment
            u_current = np.squeeze(u[cnt])
            u_current_normalized = self._normalize(u_current)

            obs, step_reward, done, info = self.env.step(u_current_normalized)
            #self.env.render()

            time += self.env.scenario.dt
            reward += step_reward

            if done:
                print(info["termination_reason"])
                break

        return obs, reward, done, info

def convertMLP(baselines_model):
    """convert a stable baselines agent to PyTorch format"""

    n_inputs = len(baselines_model.train_model.ob_space.low)
    n_actions = len(baselines_model.train_model.ac_space.low)

    torch_mlp = MultiLayerPerceptron(n_inputs=n_inputs, n_actions=n_actions)
    model_params = baselines_model.get_parameters()

    policy_keys = [key for key in model_params.keys() if "pi" in key]
    policy_params = [model_params[key] for key in policy_keys]

    for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
        param = th.from_numpy(policy_param)
        pytorch_param.data.copy_(param.data.clone().t())

    return torch_mlp


def tf2pytorch(load_path):
    model_path, vec_normalize_path = get_model_and_vecnormalize_path(load_path)
    planning_horizon = 0.8
    planner_dt = 0.4

    def env_fn():
        env = gym.make(
            "commonroad-v1",
            vehicle_params={"vehicle_model": 6},
            config_file=os.path.join(load_path, "environment_configurations.yml"),
            play=True
        )
        env = ReplanningWrapper(env, planning_horizon=planning_horizon, planner_dt=planner_dt)

        return env

    envs = DummyVecEnv([env_fn])
    env = VecNormalize.load(vec_normalize_path, envs)
    env.training = False

    model = PPO2(policy='MlpPolicy', env=env).load(load_path=model_path)

    # convert agent to PyTorch network
    agent_ = convertMLP(model)

    # test the converted model
    x = np.random.uniform(-1., 1., (1, 35))
    y, _ = model.predict(x, deterministic=True)
    y_ = agent_.forward(x)

    assert np.allclose(y, y_)

    # save the PyTorch network
    th.save(agent_.state_dict(), os.path.join(load_path, 'PyTorchModel'))

    # load normalization
    with open(vec_normalize_path, "rb") as f:
        vec_normalize = pickle.load(f)

    mean = vec_normalize.obs_rms.mean
    var = vec_normalize.obs_rms.var
    epsilon = vec_normalize.epsilon
    clip = vec_normalize.clip_obs

    with open(os.path.join(load_path, 'normalization.pkl'), 'wb') as f:
        pickle.dump([mean, var, epsilon, clip], f)

if __name__ == "__main__":
    tf2pytorch("./agent")