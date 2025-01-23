import statistics
import time
import torch
from math import exp
from .defaultconf import *


cav_agent = None
load_cav_agent_flag = True
load_pytorch_model = True
d2rl_flag = False
d2rl_criticality_threshold = 0.0
d2rl_slightlycritical = False
d2rl_subcritical_agent = None
RSS_flag = True
env_mode = "NADE"
computational_analysis_flag = False
debug_critical = False
more_info_critical = False
nade_offline_collection = False

nade_agent = None
load_nade_agent_flag = False

precise_criticality_flag = True
precise_criticality_threshold = 0.9 # 0.01, 0.9, 0.07, 0.5
precise_weight_flag = False
precise_weight_threshold = 1.0

lane_change_amplify = 1.0 # 1.0, 0.01
max_lane_change_probability = 0.99
weight_threshold = 0
epsilon_value = 0.99 # 0.5, 0.99, 1-1e-9
criticality_threshold = 0 # 1e-4
nade_criticality_threshold = 1e-4
simulation_config["map"] = "3Lane"  # "3Lane" "Mcity" "2Lane" "ACM" "2LaneLong"
experiment_config["mode"] = "NADE" # "NADE" "risk_NDE" "DRL_train" "offline"
experiment_config["lr"] = 0.00001  # 5e-5 0.0001 3e-6
simulation_config["epsilon_setting"] = "fixed" # "fixed" "drl" "varied"
simulation_config["explore_mode"] = "Normal" # "Normal" "IS" "Adaptive"
experiment_config["AV_model"] = "RLNew" #"RL" "Surrogate" "IDM" "IDM_SafetyGuard" "RL_SafetyGuard" "RLNew"
simulation_config["epsilon_type"] = "continuous"
record_video = False
frame_index = 0
def set_frame_index(new_frame_index):
    global frame_index
    frame_index = new_frame_index
simulation_config["initialization_rejection_sampling_flag"] = False
simulation_config["gui_flag"] = False
simulation_config["speed_mode"] = "high_speed" # "low_speed" "high_speed"
simulation_config["ray_ckpt_path_list"] = []
simulation_config["pytorch_model_path_list"] = []
experiment_config["experiment_name"] = "mytest"

initial_epsilon_for_DRL = 1
treesearch_config["search_depth"] = 1
treesearch_config["surrogate_model"] = "surrogate"  # "AVI" "surrogate"
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1
simulation_config["initialization_rejection_sampling_flag"] = False
experiment_config["log_mode"] = "offlinecollect"  # "all" "crash" "offlinecollect" "offlinecollect_all"
compress_log_flag = True
traffic_flow_config["BV"] = True
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 8  # [m/s2]
experiment_config["server_name"] = "" # "AW1" #"GreatLakes" "AW3" "GL_single_job_array" "rejection_parameter_est" "Omen2"


experiment_config["root_folder"] = ""
experiment_config["code_root_folder"] = ""
experiment_config["experiment_name"] = ""
experiment_config["train_flag"] = False
experiment_config["lr"] = 0.00001
experiment_config["episode_num"] = 20000
simulation_config["ray_ckpt_path_list"] = []
simulation_config["slightly_critical_ray_ckpt_path_list"] = []
simulation_config["subcritical_ray_ckpt_path"] = []
simulation_config["subcritical_ray_ckpt_num"] = 33
simulation_config["pytorch_model_path_list"] = []
simulation_config["pytorch_nade_model_path"] = "" 
simulation_config["ray_nade_model_path"] = ""

simulation_config["safetyguard_flag"] = True
simulation_config["neuralmetric_flag"] = True
simulation_config["neuralmetric_config_path"] = ""
simulation_config["metric"] = "" # "TTC", "SMAR", "WTTC"

def load_ray_agent(checkpoint_path_list, slightly_critical_ckpt=[]):
    discriminator_agent = MixedAgent(checkpoint_path_list, slightly_critical_ckpt)
    return discriminator_agent


class MixedAgent:
    def __init__(self, checkpoint_path_list, slightly_critical_ckpt=[]):
        self.agent_list = []
        
        if not load_pytorch_model:
            print("load ray model")
            import ray
            import ray.rllib.agents.ppo as ppo
            import ray.rllib.agents.dqn.apex as apex
            from ray.tune.registry import register_env
            from envs.gymenv import RL_NDE
            # from ppo_trainer_custom import MyPPOTrainer
            def env_creator(env_config):
                return RL_NDE(train_flag=False)  # return an env instance
            register_env("my_env", env_creator)
            if d2rl_flag and experiment_config["server_name"] == "GL_single_job_array":
                ray.init(
                    address=os.environ["ip_head"], include_dashboard=False, ignore_reinit_error=True)
            else:
                ray.init(local_mode=True, include_dashboard=False,
                        ignore_reinit_error=True)
            config = ppo.DEFAULT_CONFIG.copy()
            config["num_gpus"] = 0
            config["num_workers"] = 0
            config["framework"] = "torch"
            # discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
            # for ckpt_path in checkpoint_path_list:
            #     discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
            #     discriminator_agent.restore(ckpt_path)
            #     self.agent_list.append(discriminator_agent)
            if len(checkpoint_path_list) == 2:
                discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
                discriminator_agent.restore(checkpoint_path_list[0])
                self.agent_list.append(discriminator_agent)
                # config["explore"] = False
                discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
                discriminator_agent.restore(checkpoint_path_list[1])
                self.agent_list.append(discriminator_agent)
            else:
                for ckpt_path in checkpoint_path_list:
                    discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
                    discriminator_agent.restore(ckpt_path)
                    self.agent_list.append(discriminator_agent)
            
            # config["explore"] = False
            # for ckpt_path in slightly_critical_ckpt:
            #     discriminator_agent = MyPPOTrainer(config=config, env="my_env")
            #     discriminator_agent.restore(ckpt_path)
            #     self.agent_list.append(discriminator_agent)
        else:
            for path in checkpoint_path_list:
                print("load pytorch model from ",path)
                model = torch.jit.load(path)
                model.eval()
                self.agent_list.append(model)
        # test_random_agent = False
        # if test_random_agent:
        #     random_agent = ppo.PPOTrainer(config=config, env="my_env")
        #     cp_path = random_agent.save()
        #     print(cp_path)
        #     self.agent_list.append(random_agent)
        # for agent in self.agent_list:
        #     print(id(agent))

    # @profile
    def compute_action(self, observation, highly_critical=False, slightly_critical=False):
        if len(self.agent_list) == 1:
            agent = self.agent_list[0]
        elif len(self.agent_list) == 2:
            # if highly_critical or slightly_critical:
            if highly_critical:
                agent = self.agent_list[1]
            else:
                agent = self.agent_list[0]
        elif len(self.agent_list) == 3:
            if slightly_critical:
                # print("find slightly")
                agent = self.agent_list[2]
            elif highly_critical:
                # print("find highly")
                agent = self.agent_list[1]
            else:
                # print("find normal")
                agent = self.agent_list[0]
        else:
            raise ValueError("Too many/few RL agents: number=",
                             len(self.agent_list))
        # action_result = agent.compute_single_action(observation)
        if not load_pytorch_model:
            action_result = agent.compute_action(observation)
        else:
            o = torch.reshape(torch.tensor(observation), (1,len(observation)))
            out = agent({"obs":o},[torch.tensor([0.0])],torch.tensor([1]))
            action = out[0][0]
            action_result = np.array([np.clip((float(action[0])+1)*3-4,-4.,2.), np.clip((float(action[1])+1)*10-10,-10.,10.)])
        # print(action_result_list, statistics.mean(action_result_list))
        return action_result


class torch_discriminator_agent:
    def __init__(self, load_path):
        print("load nade model from", load_path)
        self.model = torch.jit.load(load_path)
        self.model.eval()

    def compute_action(self, observation):
        lb = 0.001
        ub = 0.999
        obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
        out = self.model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
        if simulation_config["epsilon_type"] == "discrete":
            action = torch.argmax(out[0][0])
        else:
            action = np.clip((float(out[0][0][0])+1)*(ub-lb)/2 + lb, lb, ub)
        return action


def load_discriminator_agent(): #mode="ray"
    if not load_pytorch_model:
        import ray
        import ray.rllib.agents.ppo as ppo
        from ray.tune.registry import register_env
        from envs.NADE_gym_env_offline import DRL_gym_ENV_offline
        def env_creator(env_config):
            return DRL_gym_ENV_offline()  # return an env instance
        register_env("my_env", env_creator)
        if experiment_config["mode"] == "DRL_train":
            ray.init(address=os.environ["ip_head"], include_dashboard=False, ignore_reinit_error=True)
        else:
            ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_gpus"] = 0
        config["num_workers"] = 0
        config["framework"] = "torch"
        config["explore"] = False
        discriminator_agent = ppo.PPOTrainer(config=config, env="my_env")
        checkpoint_path = ""
        discriminator_agent.restore(checkpoint_path)
    else:
        discriminator_agent = torch_discriminator_agent(simulation_config["pytorch_nade_model_path"])
    return discriminator_agent
