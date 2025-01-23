"""
CommonRoad Gym environment
"""
import gym

# Notice: this code is run everytime the gym_commonroad module is imported
# this might be pretty shady but seems to be common practice so let's at least catch the errors occurring here
try:
    gym.envs.register(
        id="commonroad_sumo-v0",
        entry_point="gym_commonroad_sumo.commonroad_sumo_env:CommonroadEnv_SUMO",
        kwargs=None,
    )
except gym.error.Error:
    print("[gym_commonroad/__init__.py] Error occurs while registering commonroad_sumo-v0")
    pass

try:
    gym.envs.register(
        id="commonroad_nade-v0",
        entry_point="gym_commonroad_sumo.gym_commonroad_nade:CommonRoadEnv_Nade",
        kwargs=None,
    )
except gym.error.Error:
    print(
        "[gym_commonroad/__init__.py] Error occurs while registering commonroad_sumo-v0"
    )
    pass
