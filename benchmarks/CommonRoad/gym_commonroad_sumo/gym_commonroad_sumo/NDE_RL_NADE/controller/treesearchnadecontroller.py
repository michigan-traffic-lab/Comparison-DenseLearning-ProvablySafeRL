import math
from os import times
from numpy.core.numeric import full
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp import utils
from gym_commonroad_sumo.NDE_RL_NADE.controller.nadecontroller import NADEBackgroundController
import numpy as np
from gym_commonroad_sumo.NDE_RL_NADE.controller.nddcontroller import NDDController
from gym_commonroad_sumo.NDE_RL_NADE.controller.lowspeednddcontroller import LowSpeedNDDController
import gym_commonroad_sumo.NDE_RL_NADE.conf.conf as conf
import gym_commonroad_sumo.NDE_RL_NADE.utils as utils
from math import isclose
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from collections import OrderedDict
# import torch
# import torch.nn as nn
# torch.manual_seed(0)
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# torch.set_default_tensor_type(torch.FloatTensor)

if conf.simulation_config["speed_mode"] == "low_speed":
    BASE_NDD_CONTROLLER = LowSpeedNDDController
else:
    BASE_NDD_CONTROLLER = NDDController

veh_length = 5.0
veh_width = 1.8
circle_r = 1.227
tem_len = math.sqrt(circle_r**2-(veh_width/2)**2)
LANE_WIDTH = 4.0

# @profile
def collision_check(traj1, traj2):
    time_series = list(traj1.keys())
    for time in time_series:
        center_list_1 = get_circle_center_list(traj1[time])
        center_list_2 = get_circle_center_list(traj2[time])
        for p1 in center_list_1:
            for p2 in center_list_2:
                dist = cal_dist(p1, p2)
                if dist <= 2*circle_r: # ! 09182021 more conservative on the 3_circle collision check 
                    return True
    return False


def get_circle_center_list(traj_point):
    center1 = (traj_point["x_lon"], traj_point["x_lat"])
    if "heading" in traj_point and traj_point["heading"] is not None:
        heading = (90-traj_point["heading"])/180*math.pi
    else:
        if traj_point["v_lon"] == 0:
            heading = 0
        else:
            heading = math.atan(traj_point["v_lat"]/traj_point["v_lon"])
    center0 = (
        center1[0]+(veh_length/2-tem_len)*math.cos(heading),
        center1[1]+(veh_length/2-tem_len)*math.sin(heading)
    )
    center2 = (
        center1[0]-(veh_length/2-tem_len)*math.cos(heading),
        center1[1]-(veh_length/2-tem_len)*math.sin(heading)
    )
    center_list = [center0, center1, center2]
    return center_list


def cal_dist(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


class Traj:
    def __init__(self, x_lon, x_lat, v_lon, v_lat, lane_index, acceleration=0, heading=None, lane_list_info=conf.lane_list):
        self.traj_info = {}
        self.lane_list_info = lane_list_info
        self.traj_info["0.0"] = {"x_lon": x_lon, "x_lat": x_lat,
                                 "v_lon": v_lon, "v_lat": v_lat, "acceleration": acceleration, "lane_index": lane_index, "heading": heading}

    def crop(self, start_time, end_time):
        pop_key_list = []
        for time in self.traj_info:
            if (float(time) < start_time and not isclose(start_time, float(time))) or (float(time) > end_time and not isclose(end_time, float(time))):
                pop_key_list.append(time)
        for key in pop_key_list:
            self.traj_info.pop(key)

    def predict_with_action(self, action, start_time=0.0, time_duration=1.0):
        if action == "left" or action == "right" or action == "still":
            new_action = {"lateral": action, "longitudinal": 0}
            if action == "still":
                new_action["lateral"] = "central"
            action = new_action
        ini_traj_info = self.traj_info["%.1f" % start_time]
        # if time_duration != 1:
        #     raise ValueError("not supported time duration in trajectory prediction")
        predict_traj_info = {}
        if action["lateral"] == "left":
            predict_traj_info["lane_index"] = min(ini_traj_info["lane_index"] + 1, len(self.lane_list_info)-1)
            predict_traj_info["x_lat"] = min(ini_traj_info["x_lat"] + 4.0, self.lane_list_info[-1]+LANE_WIDTH/2)
        elif action["lateral"] == "right":
            predict_traj_info["lane_index"] = max(ini_traj_info["lane_index"] - 1, 0)
            predict_traj_info["x_lat"] = max(ini_traj_info["x_lat"] - 4.0, self.lane_list_info[0]-LANE_WIDTH/2)
        else:
            predict_traj_info["lane_index"] = ini_traj_info["lane_index"]
            predict_traj_info["x_lat"] = ini_traj_info["x_lat"]
        acceleration = action["longitudinal"]
        # predict_traj_info["x_lon"] = ini_traj_info["x_lon"] + \
        #     ini_traj_info["v_lon"]*time_duration + \
        #     0.5 * acceleration * time_duration**2
        predict_traj_info["x_lon"] = ini_traj_info["x_lon"] + \
            ini_traj_info["v_lon"]*time_duration # ! diable the second order prediction of position (acceleration) while predicting lane change maneuvers 09182021
        predict_traj_info["v_lon"] = ini_traj_info["v_lon"] + \
            acceleration * time_duration
        predict_traj_info["v_lat"] = 0
        predict_traj_info["acceleration"] = ini_traj_info["acceleration"]
        self.traj_info["%.1f" %
                       (start_time + time_duration)] = predict_traj_info
        self.interpolate(start_time=start_time,
                         end_time=start_time+time_duration)
        return self.traj_info["%.1f" % (start_time + time_duration)]

    def predict_without_action(self, start_time=0.0, time_duration=1.0, is_lane_change=None):
        ini_traj_info = self.traj_info["%.1f" % start_time]
        if time_duration == 1.0:
            predict_traj_info = {}
            if is_lane_change is None:
                # if ini_traj_info["v_lat"] > 0.5 and not ini_traj_info["x_lat"] >= max(self.lane_list_info):
                #     predict_traj_info["lane_index"] = np.argwhere(
                #         self.lane_list_info > ini_traj_info["x_lat"])[0].item()
                # elif ini_traj_info["v_lat"] < -0.5 and not ini_traj_info["x_lat"] <= min(TreeSearchNADEBackgroundController.lane_list):
                #     predict_traj_info["lane_index"] = np.argwhere(
                #         self.lane_list_info < ini_traj_info["x_lat"])[-1].item()
                # else:
                if 1:
                    predict_traj_info["lane_index"] = ini_traj_info["lane_index"]
            else:
                lane_offset = ini_traj_info["x_lat"] - self.lane_list_info[ini_traj_info["lane_index"]]
                # add 1 cm resolution
                if abs(lane_offset) <= 0.01:
                    lane_offset = 0.0
                if is_lane_change and ini_traj_info["v_lat"] > 0.5 and not ini_traj_info["x_lat"] >= max(self.lane_list_info) and ini_traj_info["v_lat"]*lane_offset > 0:
                    predict_traj_info["lane_index"] = np.argwhere(
                        self.lane_list_info > ini_traj_info["x_lat"])[0].item()
                elif is_lane_change and ini_traj_info["v_lat"] < -0.5 and not ini_traj_info["x_lat"] <= min(TreeSearchNADEBackgroundController.lane_list) and ini_traj_info["v_lat"]*lane_offset > 0:
                    predict_traj_info["lane_index"] = np.argwhere(
                        self.lane_list_info < ini_traj_info["x_lat"])[-1].item()
                else:
                    predict_traj_info["lane_index"] = ini_traj_info["lane_index"]
            new_x_lat = self.lane_list_info[int(
                predict_traj_info["lane_index"])]
            if isclose(ini_traj_info["v_lat"], 0.0):
                estimated_lane_change_time = 100
            else:
                estimated_lane_change_time = round(
                    abs((new_x_lat - ini_traj_info["x_lat"])/ini_traj_info["v_lat"]), 1)
            if not is_lane_change or estimated_lane_change_time > 1:
                predict_traj_info["x_lat"] = self.lane_list_info[int(
                    predict_traj_info["lane_index"])]
                predict_traj_info["x_lon"] = ini_traj_info["x_lon"] + \
                    ini_traj_info["v_lon"]*time_duration + 0.5 * ini_traj_info["acceleration"] * (time_duration**2)
                predict_traj_info["v_lat"] = 0
                predict_traj_info["v_lon"] = ini_traj_info["v_lon"] + ini_traj_info["acceleration"] * time_duration
                predict_traj_info["acceleration"] = ini_traj_info["acceleration"]
                self.traj_info["%.1f" %
                               (start_time + time_duration)] = predict_traj_info
                self.interpolate(start_time=start_time,
                                 end_time=start_time+time_duration)
            else:
                predict_traj_info["x_lat"] = self.lane_list_info[int(
                    predict_traj_info["lane_index"])]
                predict_traj_info["x_lon"] = ini_traj_info["x_lon"] + \
                    ini_traj_info["v_lon"]*estimated_lane_change_time
                predict_traj_info["v_lat"] = 0
                predict_traj_info["v_lon"] = ini_traj_info["v_lon"]
                predict_traj_info["acceleration"] = ini_traj_info["acceleration"]
                self.traj_info["%.1f" % (
                    start_time + estimated_lane_change_time)] = predict_traj_info
                # print(estimated_lane_change_time)
                # print(is_lane_change, ini_traj_info["v_lat"])
                self.predict_without_action(
                    start_time + estimated_lane_change_time, 1-estimated_lane_change_time)
                self.interpolate(start_time=start_time,
                                 end_time=start_time+estimated_lane_change_time)
                self.interpolate(
                    start_time=start_time+estimated_lane_change_time, end_time=start_time + time_duration)
        else:
            self.predict_without_action(start_time, 1.0, is_lane_change)
            self.interpolate(start_time=start_time,
                             end_time=start_time+time_duration)
        return self.traj_info["%.1f" % (start_time + time_duration)]

    def interpolate(self, start_time, end_time, time_resolution=0.1):
        ini_info = self.traj_info["%.1f" % start_time]
        end_info = self.traj_info["%.1f" % end_time]
        if ("%.1f" % start_time) in self.traj_info and ("%.1f" % end_time) in self.traj_info:
            inter_list = drange(start_time, end_time, 0.1)
            for time_value in inter_list:
                if isclose(time_value, start_time) or isclose(time_value, end_time):
                    continue
                else:
                    start_ratio = (end_time - time_value) / \
                        (end_time - start_time)
                    tmp_traj = {}
                    tmp_traj["x_lon"] = start_ratio * ini_info["x_lon"] + \
                        (1-start_ratio) * end_info["x_lon"]
                    tmp_traj["x_lat"] = start_ratio * ini_info["x_lat"] + \
                        (1-start_ratio) * end_info["x_lat"]
                    tmp_traj["v_lon"] = start_ratio * ini_info["v_lon"] + \
                        (1-start_ratio) * end_info["v_lon"]
                    tmp_traj["v_lat"] = start_ratio * ini_info["v_lat"] + \
                        (1-start_ratio) * end_info["v_lat"]
                    tmp_traj["lane_index"] = (
                        np.abs(self.lane_list_info - tmp_traj["x_lat"])).argmin().item()
                    tmp_traj["acceleration"] = ini_info["acceleration"]
                    self.traj_info["%.1f" % time_value] = tmp_traj
        else:
            raise ValueError("Interpolate between non-existing points")


class TreeSearchNADEBackgroundController(NADEBackgroundController):
    """Tree Search Based NADE controller

    Args:
        NADEBackgroundController (controller): the original rule-based controller
    """
    MAX_TREE_SEARCH_DEPTH = conf.treesearch_config["search_depth"]
    ACTION_NUM = 33  # full actions
    ACTION_TYPE = {"left": 0, "right": 1, "still": list(range(2, 33))}
    input_lower_bound = [-50, 20, 0] * 9
    input_lower_bound[0] = 400
    input_upper_bound = [50, 40, 2] * 9
    input_upper_bound[0] = 800
    input_lower_bound = np.array(input_lower_bound)
    input_upper_bound = np.array(input_upper_bound)
    if conf.treesearch_config["surrogate_model"] == "surrogate":
        SURROGATE_MODEL_FUNCTION = utils._get_Surrogate_CAV_action_probability
    elif conf.treesearch_config["surrogate_model"] == "AVI":
        SURROGATE_MODEL_FUNCTION = IDMController.decision_pdf
    PREDICT_MODEL_FUNCTION = IDMController.decision
    # lane_list = np.array([42.0, 46.0, 50.0])
    lane_list = conf.lane_list

    def update_single_vehicle_obs_no_action(veh, duration=1.0, is_lane_change=None, acceleration=None):
        if acceleration is None:
            acceleration = veh["acceleration"]
        initial_traj = Traj(veh["position"][0], veh["position"][1],
                            veh["velocity"], veh["lateral_speed"], veh["lane_index"], acceleration, veh["heading"])
        if is_lane_change is None:
            is_lane_change = utils.is_lane_change(veh)
        new_traj_result = initial_traj.predict_without_action(
            0.0, duration, is_lane_change)
        # new_veh = ujson.loads(ujson.dumps(veh))
        new_veh = dict(veh)
        new_veh["position"] = (new_traj_result["x_lon"],
                               new_traj_result["x_lat"])
        new_veh["velocity"] = new_traj_result["v_lon"]
        new_veh["lateral_velocity"] = new_traj_result["v_lat"]
        new_veh["lane_index"] = new_traj_result["lane_index"]
        if new_veh["lane_index"] == 0:
            new_veh["could_drive_adjacent_lane_left"] = True
            new_veh["could_drive_adjacent_lane_right"] = False
        if new_veh["lane_index"] == len(conf.lane_list) -1:
            new_veh["could_drive_adjacent_lane_left"] = False
            new_veh["could_drive_adjacent_lane_right"] = True
        return new_veh, initial_traj

    @staticmethod
    # @profile
    def is_CF(cav_obs, bv_obs):
        """Check whether the given two vehicle is following each other

        Args:
            CAV (dict): CAV Ego observation
            BV (dict): BV Ego observation
            cav_obs (dict): CAV total observation
            bv_obs (dict): BV total observation

        Returns:
            str: the CF condition of CAV and BV
        """
        CF_info = False
        bv_r1 = bv_obs['Foll']
        bv_f1 = bv_obs["Lead"]
        CAV_id = cav_obs["Ego"]["veh_id"]
        bv_v, bv_range_CAV, bv_rangerate_CAV = None, None, None
        if bv_r1 is not None and bv_r1['veh_id'] == CAV_id:
            CF_info = "CAV_BV"  # CAV is following BV
            bv_v, bv_range_CAV, bv_rangerate_CAV = bv_obs["Ego"]["velocity"], bv_obs["Ego"]["position"][0] - cav_obs["Ego"]["position"][0] - conf.LENGTH, bv_obs["Ego"]["velocity"] - cav_obs["Ego"]["velocity"]
        if bv_f1 is not None and bv_f1['veh_id'] == CAV_id:
            CF_info = "BV_CAV"  # BV is following CAV
            bv_v, bv_range_CAV, bv_rangerate_CAV = bv_obs["Ego"]["velocity"], cav_obs["Ego"]["position"][0] - bv_obs["Ego"]["position"][0] - conf.LENGTH, cav_obs["Ego"]["velocity"] - bv_obs["Ego"]["velocity"]
        return CF_info, bv_v, bv_range_CAV, bv_rangerate_CAV 

    @staticmethod
    # @profile
    def update_single_vehicle_obs(veh, action, duration=conf.simulation_resolution):
        new_pos_x, new_pos_y = 0, 0
        # new_veh = ujson.loads(ujson.dumps(veh))
        new_veh = dict(veh)
        new_veh["velocity"] = veh["velocity"]
        new_pos_x = veh["position"][0] + veh["velocity"]*duration
        # if action == "left":
        if (action == "left" or action == "right") and not utils.is_lane_change(veh):
            traj = Traj(veh["position"][0], veh["position"][1],
                        veh["velocity"], veh["lateral_speed"], veh["lane_index"], veh["acceleration"], veh["heading"])
            new_traj_result = traj.predict_with_action(action, 0.0, duration)
            traj.crop(start_time=0.0, end_time=duration)
            # new_veh = ujson.loads(ujson.dumps(veh))
            new_veh = dict(veh)
            new_veh["position"] = (
                new_traj_result["x_lon"], new_traj_result["x_lat"])
            new_veh["velocity"] = new_traj_result["v_lon"]
            new_veh["lateral_velocity"] = new_traj_result["v_lat"]
            new_veh["lane_index"] = new_traj_result["lane_index"]
            if new_veh["lane_index"] == 0:
                new_veh["could_drive_adjacent_lane_left"] = True
                new_veh["could_drive_adjacent_lane_right"] = False
            if new_veh["lane_index"] == len(conf.lane_list) -1:
                new_veh["could_drive_adjacent_lane_left"] = False
                new_veh["could_drive_adjacent_lane_right"] = True
        else:
            if action == "left" or action == "right":
                action = "still"
            acceleration_mode_dict = {"brake": -4.0, "accelerate":2.0, "still": veh["acceleration"], "constant":0}
            acceleration = acceleration_mode_dict[action]
            new_veh, traj = TreeSearchNADEBackgroundController.update_single_vehicle_obs_no_action(
                veh, duration, acceleration=acceleration)
        return new_veh, traj
    

    @staticmethod
    # @profile
    def update_single_cav_obs_nn(veh, prev_full_obs, full_traj, duration=conf.simulation_resolution):
        total_bv_num = 8
        cav_history_traj_dict = prev_full_obs["CAV"]["history"]
        if len(cav_history_traj_dict) < 10:
            return TreeSearchNADEBackgroundController.update_single_cav_obs_model(veh, prev_full_obs, full_traj, duration=duration)
        cav_history_keys = list(cav_history_traj_dict.keys())
        cav_history_keys.sort(key=lambda x:float(x))
        cav_history_keys = cav_history_keys[-10:]
        cav_center_x, cav_center_y = cav_history_traj_dict[cav_history_keys[-1]
                                                           ][0], cav_history_traj_dict[cav_history_keys[-1]][1]
        if set(prev_full_obs.keys())-set(["CAV"]) != set(full_traj.keys()):
            raise ValueError("Traj and history not match!")
        bv_traj_list = []
        bv_candidates = list(full_traj.keys())
        bv_candidates.extend([None] * (total_bv_num - len(bv_candidates)))
        for bv_id in bv_candidates:
            if bv_id is None:
                bv_traj_list.extend([-100]*40)
                continue
            bv_traj_history_tmp = prev_full_obs[bv_id]["history"]
            bv_traj_predict_tmp = full_traj[bv_id].traj_info
            history_keys = list(bv_traj_history_tmp.keys())
            predict_keys = list(set(bv_traj_predict_tmp.keys()) - set(["0.0"]))
            history_keys.sort(key=lambda x:float(x))
            predict_keys.sort(key=lambda x:float(x))
            if len(history_keys) < 10:
                return TreeSearchNADEBackgroundController.update_single_cav_obs_model(veh, prev_full_obs, full_traj, duration=duration)
            history_keys = history_keys[-10:]
            predict_keys = predict_keys[:10]
            bv_traj_tmp = []
            for key in history_keys:
                bv_traj_tmp.extend(
                    [bv_traj_history_tmp[key][0]-cav_center_x, bv_traj_history_tmp[key][1]-cav_center_y])
            for key in predict_keys:
                bv_traj_tmp.extend([bv_traj_predict_tmp[key]["x_lon"]-cav_center_x,
                                    bv_traj_predict_tmp[key]["x_lat"]-cav_center_y])
            bv_traj_list.extend(bv_traj_tmp)

        cav_traj_list = []
        cav_history_keys = list(cav_history_traj_dict.keys())
        cav_history_keys.sort(key=lambda x:float(x))
        cav_history_keys = cav_history_keys[-10:]
        for key in cav_history_keys:
            cav_traj_list.extend(
                [cav_history_traj_dict[key][0]-cav_center_x, cav_history_traj_dict[key][1]-cav_center_y])
        input_list = cav_traj_list
        input_list.extend(bv_traj_list)
        input_array = np.array(input_list)

        input_lower_bound = np.array([-120, -10] * 170)
        input_upper_bound = np.array([120, 10] * 170)
        input_array_normalized = (input_array-input_lower_bound) / \
            (input_upper_bound-input_lower_bound).reshape(1, -1)
        input_array_normalized_tensor = torch.tensor(
            input_array_normalized, dtype=torch.float32)
        output_tensor = conf.net_G(
            input_array_normalized_tensor.float().to(device))
        output_array = output_tensor.detach().numpy()
        # output_offset = np.array([cav_center_x, cav_center_y] * 10)
        output_offset = np.array([veh["position"][0], veh["position"][1]] * 10)
        final_output_np = (output_array+output_offset).flatten()
        predict_keys = ["0.1", "0.2", "0.3", "0.4",
                        "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
        cav_ini_traj = Traj(veh["position"][0], veh["position"][1],
                            veh["velocity"], veh["lateral_speed"], veh["lane_index"], veh["acceleration"], veh["heading"])
        full_traj["CAV"] = cav_ini_traj
        for i in range(len(predict_keys)):
            key = predict_keys[i]
            x, y = final_output_np[2*i], final_output_np[2*i+1]
            cav_ini_traj.traj_info[key] = {
                "x_lon": x, "x_lat": y, "v_lon": veh["velocity"], "v_lat": 0}
            cav_ini_traj.traj_info[key]["lane_index"] = (
                np.abs(cav_ini_traj.lane_list_info - cav_ini_traj.traj_info[key]["x_lat"])).argmin().item()
        a = TreeSearchNADEBackgroundController.traj_to_obs(
            prev_full_obs, full_traj, "1.0")
        cav_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
            a, "CAV")
        if cav_obs["Ego"]["lane_index"] == 0:
            cav_obs["Ego"]["could_drive_adjacent_lane_left"] = True
            cav_obs["Ego"]["could_drive_adjacent_lane_right"] = False
        if cav_obs["Ego"]["lane_index"] == len(conf.lane_list) -1:
            cav_obs["Ego"]["could_drive_adjacent_lane_left"] = False
            cav_obs["Ego"]["could_drive_adjacent_lane_right"] = True
        return dict(cav_obs["Ego"]), cav_ini_traj

    @staticmethod
    def update_single_cav_obs_model(veh, prev_full_obs, full_traj, duration=conf.simulation_resolution):
        if abs(veh["lateral_speed"]) > 0.5 and abs(veh["lateral_offset"]) > 0.1:
            cav_ego_obs, cav_ini_traj = TreeSearchNADEBackgroundController.update_single_vehicle_obs_no_action(
                veh, 1.0, True)
            time_steps = list(cav_ini_traj.traj_info.keys())
            time_steps.sort(key=lambda x:float(x))
            for time in time_steps:
                if isclose(cav_ini_traj.traj_info[time]["v_lat"], 0):
                    start_time = float(time)
                    break
        else:
            # ! If AV in the lane change mode, then apply no action in prediction
            cav_ini_traj = Traj(veh["position"][0], veh["position"][1],
                                veh["velocity"], veh["lateral_speed"], veh["lane_index"], veh["acceleration"], veh["heading"])
            start_time = 0.0
        if start_time >= duration or isclose(start_time, duration):
            raise ValueError(
                "Predict the lane change maneuver larger than 1s, wrong")
        full_traj["CAV"] = cav_ini_traj
        for veh_id in full_traj:
            traj_tmp = full_traj[veh_id].traj_info
            time_steps = list(traj_tmp.keys())
            time_steps.sort(key=lambda x:float(x))
            break
        for timestep in time_steps:
            if float(timestep) < start_time and not isclose(float(timestep), start_time):
                continue
            if isclose(float(timestep), float(duration)):
                break
            a = TreeSearchNADEBackgroundController.traj_to_obs(
                prev_full_obs, full_traj, timestep)
            cav_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
                a, "CAV")
            cav_action, cav_mode = TreeSearchNADEBackgroundController.PREDICT_MODEL_FUNCTION(
                cav_obs)
            if cav_action["lateral"] == "central":
                cav_ini_traj.predict_with_action(
                    cav_action, start_time=float(timestep), time_duration=0.1)
            else:
                cav_ini_traj.predict_with_action(cav_action, start_time=float(
                    timestep), time_duration=1-float(timestep))
        if cav_obs["Ego"]["lane_index"] == 0:
            cav_obs["Ego"]["could_drive_adjacent_lane_left"] = True
            cav_obs["Ego"]["could_drive_adjacent_lane_right"] = False
        if cav_obs["Ego"]["lane_index"] == len(conf.lane_list) -1:
            cav_obs["Ego"]["could_drive_adjacent_lane_left"] = False
            cav_obs["Ego"]["could_drive_adjacent_lane_right"] = True
        return dict(cav_obs["Ego"]), cav_ini_traj

    @staticmethod
    # @profile
    def traj_to_obs(prev_full_obs, full_traj, time):
        obs = {}
        for key in prev_full_obs:
            obs[key] = dict(prev_full_obs[key])
        for veh_id in full_traj:
            obs[veh_id]["position"] = (
                full_traj[veh_id].traj_info[time]["x_lon"], full_traj[veh_id].traj_info[time]["x_lat"])
            obs[veh_id]["velocity"] = full_traj[veh_id].traj_info[time]["v_lon"]
            obs[veh_id]["lateral_velocity"] = full_traj[veh_id].traj_info[time]["v_lat"]
            obs[veh_id]["lane_index"] = full_traj[veh_id].traj_info[time]["lane_index"]
        return obs

    @staticmethod
    # @profile
    def update_obs(full_obs, cav_id, bv_id, bv_action, predicted_full_obs=None, predicted_full_traj=None, predicted_cav_action="still"):
        new_full_obs = {}
        for key in full_obs:
            new_full_obs[key] = dict(full_obs[key])
        trajectory_obs = {} 
        for veh_id in full_obs:
            action = "still"
            if veh_id == bv_id:
                action = bv_action
            if veh_id == cav_id:
                continue
            if action:
                vehicle = new_full_obs[veh_id]
                if predicted_full_obs is None or predicted_full_traj is None:
                    new_full_obs[veh_id], trajectory_obs[veh_id] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(
                    vehicle, action)
                else:
                    new_full_obs[veh_id] = predicted_full_obs[veh_id][action]
                    trajectory_obs[veh_id] = predicted_full_traj[veh_id][action]
        # new_full_obs[cav_id], trajectory_obs[cav_id] = TreeSearchNADEBackgroundController.update_single_cav_obs_model(
            # new_full_obs[cav_id], full_obs, trajectory_obs)
        # new_full_obs[cav_id], trajectory_obs[cav_id] = TreeSearchNADEBackgroundController.update_single_cav_obs_nn(
        #     new_full_obs[cav_id], full_obs, trajectory_obs)
        new_full_obs[cav_id], trajectory_obs[cav_id] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(
                    new_full_obs[cav_id], predicted_cav_action) 

        # Sort the observation using the distance from the CAV
        av_pos = new_full_obs["CAV"]["position"]
        for veh_id in new_full_obs:
            bv_pos = new_full_obs[veh_id]["position"]
            new_full_obs[veh_id]["euler_distance"] = utils.cal_euclidean_dist(
                av_pos, bv_pos)
        new_full_obs = OrderedDict(
            sorted(new_full_obs.items(), key=lambda item: item[1]['euler_distance']))
        for traj in trajectory_obs:
            trajectory_obs[traj].crop(0.0, 1.0)
        return new_full_obs, trajectory_obs

    @staticmethod
    # @profile
    def cav_bv_obs_to_full_obs(cav_obs, bv_obs):
        """change from cav bv observation to full observation

        Args:
            cav_obs (dict): the observation information of cav
            bv_obs (dict): the observation information of bv

        Returns:
            dict: vehicle information that contains all vehicles in cav obs and bv obs
        """
        full_obs = {}
        for cav_observe_info in cav_obs:
            if cav_obs[cav_observe_info] is None:
                continue
            observed_id = cav_obs[cav_observe_info]["veh_id"]
            if observed_id not in full_obs:
                full_obs[observed_id] = cav_obs[cav_observe_info]
        for bv_observe_info in bv_obs:
            if bv_obs[bv_observe_info] is None:
                continue
            observed_id = bv_obs[bv_observe_info]["veh_id"]
            if observed_id not in full_obs:
                full_obs[observed_id] = bv_obs[bv_observe_info]
        return full_obs

    @staticmethod
    # @profile
    def full_obs_to_cav_bv_obs(full_obs, cav_id, bv_id):
        new_full_obs = {}
        for key in full_obs:
            new_full_obs[key] = dict(full_obs[key])
        cav_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
            new_full_obs, cav_id)
        bv_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
            new_full_obs, bv_id)
        return cav_obs, bv_obs

    @staticmethod
    # @profile
    def _process_info(full_obs, ego_id=None, longi=1, lateral=0):
        ego_length = 5
        ego_lane_index = full_obs[ego_id]["lane_index"]
        ego_lane_pos = full_obs[ego_id]["position"][0]
        cand_id = None
        cand_dist = 0
        for bv_id in full_obs:
            if bv_id != ego_id:
                bv_length = 5
                bv_lane_index = full_obs[bv_id]["lane_index"]
                bv_lane_pos = full_obs[bv_id]["position"][0]
                if bv_lane_index == ego_lane_index+lateral and longi*(bv_lane_pos-ego_lane_pos) >= 0:
                    dist = abs(bv_lane_pos-ego_lane_pos)
                    if longi == 1:
                        dist -= ego_length
                    if longi == -1:
                        dist -= bv_length
                    if not cand_id:
                        cand_id = bv_id
                        cand_dist = dist
                    elif cand_dist > dist:
                        cand_id = bv_id
                        cand_dist = dist
        if cand_id is None:
            veh = None
        else:
            veh = full_obs[cand_id]
            veh["distance"] = cand_dist
        return veh

    @staticmethod
    # @profile
    def full_obs_to_single_obs(full_obs, veh_id):
        obs = {"Ego": full_obs[veh_id]}
        obs["Lead"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=1,lateral=0)
        obs["LeftLead"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=1,lateral=1)
        obs["RightLead"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=1,lateral=-1)
        obs["Foll"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=-1,lateral=0)
        obs["LeftFoll"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=-1,lateral=1)
        obs["RightFoll"] = TreeSearchNADEBackgroundController._process_info(full_obs,veh_id,longi=-1,lateral=-1)
        return obs

    @staticmethod
    # @profile
    def crash_check(cav_obs, bv_obs, cav_id, bv_id, previous_obs, traj):
        if traj is None:
            return False
        cav_traj = traj[cav_id]
        bv_traj = traj[bv_id]
        return collision_check(cav_traj.traj_info, bv_traj.traj_info)

    @staticmethod
    # @profile
    def leaf_node_check(full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth):
        """check whether the serach can be terminated

        Args:
            CAV (vehicle.observation.information["Ego"]): CAV observation information
            BV (vehicle.observation.information["Ego"]): the vehicle observation
            all_candidates (observation list): all NADE candidates for NADE decision
            CAV_action (int): the action of CAV
            BV_action (int): the action of BV
            search_depth (int): current search depth

        Returns:
            [type]: [description]
        """
        challenge_array = np.zeros(
            TreeSearchNADEBackgroundController.ACTION_NUM)
        depth_flag = (search_depth == 
                        TreeSearchNADEBackgroundController.MAX_TREE_SEARCH_DEPTH)
        CF_flag = TreeSearchNADEBackgroundController.is_CF(cav_obs, bv_obs)[0]
        bv_id = bv_obs["Ego"]["veh_id"]
        if CF_flag:
            challenge_array[2:] = TreeSearchNADEBackgroundController.get_CF_challenge_array(
                cav_obs, bv_obs)
            challenge_array[0], challenge_array[1] = np.mean(challenge_array[2:]), np.mean(challenge_array[2:]) # add challenge prediction for lane change maneuvers under the CF result
        crash_flag = TreeSearchNADEBackgroundController.crash_check(
            cav_obs, bv_obs, cav_id, bv_id, previous_obs, traj)
        if crash_flag:
            challenge_array = np.ones(
                TreeSearchNADEBackgroundController.ACTION_NUM)
        # if reach the depth and other result didn't give us the challenge array
        if depth_flag and not (CF_flag or crash_flag) and conf.treesearch_config["offline_leaf_evaluation"]:
            # print("COMING INTO NN EVALUATION!!!!!!")
            obs_for_nn, bv_order_idx = TreeSearchNADEBackgroundController.get_nn_obs_from_full(
                full_obs, bv_id)
            # print("GET OBS FOR NN!!!!!!")
            log_challenge_prediction = conf.net_G(
                torch.tensor(obs_for_nn).float().to(device))
            # print("NN PREDICTION!!!!!!")

            challenge_prediction = torch.pow(10, log_challenge_prediction-15)
            # print("NN POWER PREDICTION!!!!!!")
            challenge_estimation = challenge_prediction[0, 3*(bv_order_idx-1):3*(
                bv_order_idx-1) + 3] * conf.treesearch_config["offline_discount_factor"]
            challenge_array[0] = challenge_estimation[0].item()
            challenge_array[1] = challenge_estimation[1].item()
            challenge_array[2:] = challenge_estimation[2].item()
            # print("GOING OUT OF NN EVALUATION!!!!!!")
        return depth_flag or crash_flag, challenge_array

    @staticmethod
    # @profile
    def get_nn_obs_from_full(full_obs, bv_id):
        nn_obs = []
        state_size = 27
        cav_pos = full_obs["CAV"]["position"][0]
        cav_velocity = full_obs["CAV"]["velocity"]
        cav_lane_index = full_obs["CAV"]["lane_index"]
        nn_obs.extend([cav_pos, cav_velocity, cav_lane_index])
        i = 0
        for veh_id in full_obs:
            if veh_id == bv_id:
                bv_order_idx = i
            i = i + 1
            veh_info = full_obs[veh_id]
            veh_pos = veh_info["position"][0]
            veh_velocity = veh_info["velocity"]
            veh_lane_idx = veh_info["lane_index"]
            if len(nn_obs) < state_size:
                nn_obs.extend([veh_pos - cav_pos, veh_velocity, veh_lane_idx])
            else:
                break
        if len(nn_obs) < state_size:
            nn_obs.extend([-1]*(state_size-len(nn_obs)))
        nn_obs = (nn_obs - TreeSearchNADEBackgroundController.input_lower_bound) / \
            (TreeSearchNADEBackgroundController.input_upper_bound -
             TreeSearchNADEBackgroundController.input_lower_bound)
        return np.array(nn_obs).reshape(1, -1), bv_order_idx

    @staticmethod
    # @profile
    def tree_search_maneuver_challenge(full_obs, previous_obs, traj, cav_id, bv_id, search_depth, cav_obs=None, bv_obs=None, predicted_full_obs=None, predicted_full_traj=None):
        """generate the maneuver challenge value for a CAV and BV pair, CAV, BV action given

        Args:
            full_obs: all the controlled BV candidates
            cav_id (int): the CAV id
            bv_id (int): the BV id that will be controlled
            search_depth (int): the depth of the tree

        Returns:
            float: challenge(the prob of having crashes)
        """
        challenge_array = np.zeros(
            TreeSearchNADEBackgroundController.ACTION_NUM)
        if (not cav_obs) or (not bv_obs):
        # if 1:
            cav_obs, bv_obs = TreeSearchNADEBackgroundController.full_obs_to_cav_bv_obs(
                full_obs, cav_id, bv_id)
        leaf_flag, leaf_challenge_array = TreeSearchNADEBackgroundController.leaf_node_check(
            full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth)
        cav_action_dict, bv_action_dict, cav_pdf, bv_pdf = TreeSearchNADEBackgroundController.get_cav_bv_pdf(
            cav_obs, bv_obs)
        if leaf_flag:
            return leaf_challenge_array, bv_pdf
        else:
            # create the maneuver challenge estimation
            for bv_action in bv_action_dict:
                if bv_action_dict[bv_action] == 0:
                    continue
                else:
                    for action in ["brake", "still", "constant", "accelerate"]:
                        updated_full_obs, trajectory_obs = TreeSearchNADEBackgroundController.update_obs(
                            full_obs, cav_id, bv_id, bv_action, predicted_full_obs, predicted_full_traj, predicted_cav_action=action)
                        new_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
                            updated_full_obs, full_obs, trajectory_obs, cav_id, bv_id, search_depth + 1)
                        discount_factor = 1
                        if search_depth != 0:
                            discount_factor = conf.treesearch_config["treesearch_discount_factor"]
                        challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]] = np.maximum(discount_factor * np.sum(new_challenge_array*updated_bv_pdf), challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]])

                    updated_full_obs, trajectory_obs = TreeSearchNADEBackgroundController.update_obs(
                        full_obs, cav_id, bv_id, bv_action, predicted_full_obs, predicted_full_traj)
        return challenge_array, bv_pdf

    # @staticmethod
    # # @profile
    # def tree_search_maneuver_challenge_conservative(full_obs, previous_obs, traj, cav_id, bv_id, search_depth, cav_obs=None, bv_obs=None, predicted_full_obs=None, predicted_full_traj=None):
    #     """generate the maneuver challenge value for a CAV and BV pair, CAV, BV action given

    #     Args:
    #         full_obs: all the controlled BV candidates
    #         cav_id (int): the CAV id
    #         bv_id (int): the BV id that will be controlled
    #         search_depth (int): the depth of the tree

    #     Returns:
    #         float: challenge(the prob of having crashes)
    #     """
    #     challenge_array = np.zeros(
    #         TreeSearchNADEBackgroundController.ACTION_NUM)
    #     if (not cav_obs) or (not bv_obs):
    #     # if 1:
    #         cav_obs, bv_obs = TreeSearchNADEBackgroundController.full_obs_to_cav_bv_obs(
    #             full_obs, cav_id, bv_id)
    #     leaf_flag, leaf_challenge_array = TreeSearchNADEBackgroundController.leaf_node_check(
    #         full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth)
    #     cav_action_dict, bv_action_dict, cav_pdf, bv_pdf = TreeSearchNADEBackgroundController.get_cav_bv_pdf(
    #         cav_obs, bv_obs)
    #     if leaf_flag:
    #         return leaf_challenge_array, bv_pdf
    #     else:
    #         # create the maneuver challenge estimation
    #         for bv_action in bv_action_dict:
    #             if bv_action_dict[bv_action] == 0:
    #                 continue
    #             # if bv_action == "still" and TreeSearchNADEBackgroundController.is_CF(cav_obs, bv_obs)[0]:
    #             #     challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]] += TreeSearchNADEBackgroundController.get_CF_challenge_array(cav_obs, bv_obs) * cav_action_dict["still"]
    #             else:
    #                 for a in cav_action_dict.keys():
    #                     if a != "still":
    #                         action_poss = cav_action_dict[a]
    #                         if action_poss > 0:
    #                             updated_full_obs, trajectory_obs = TreeSearchNADEBackgroundController.update_obs(
    #                                 full_obs, cav_id, bv_id, bv_action, predicted_full_obs, predicted_full_traj, predicted_cav_action=a)
    #                             new_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
    #                                 updated_full_obs, full_obs, trajectory_obs, cav_id, bv_id, search_depth + 1)
    #                             discount_factor = 1
    #                             if search_depth != 0:
    #                                 discount_factor = conf.treesearch_config["treesearch_discount_factor"]
    #                             challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]] += discount_factor * np.sum(new_challenge_array*updated_bv_pdf) * action_poss
    #                             # test lane conflict crash
    #                             # if a != "still" and np.sum(new_challenge_array*updated_bv_pdf) > 0 and cav_obs["Ego"]["lane_index"] == 0 and bv_obs["Ego"]["lane_index"] == 2 and updated_bv_pdf[1] > 0:
    #                             #     print("FIND")
    #                         else:
    #                             continue
    #                     else:
    #                         # for action in ["brake", "accelerate", "still"]:
    #                         for action in ["brake", "still", "constant"]:
    #                             action_poss = cav_action_dict[a]
    #                             if action_poss > 0:
    #                                 updated_full_obs, trajectory_obs = TreeSearchNADEBackgroundController.update_obs(
    #                                     full_obs, cav_id, bv_id, bv_action, predicted_full_obs, predicted_full_traj, predicted_cav_action=action)
    #                                 new_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
    #                                     updated_full_obs, full_obs, trajectory_obs, cav_id, bv_id, search_depth + 1)
    #                                 discount_factor = 1
    #                                 if search_depth != 0:
    #                                     discount_factor = conf.treesearch_config["treesearch_discount_factor"]
    #                                 challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]] = np.maximum(discount_factor * np.sum(new_challenge_array*updated_bv_pdf) * action_poss, challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]])
    #                             else:
    #                                 continue        
        
    #     return challenge_array, bv_pdf

    @staticmethod
    # @profile
    def get_CF_challenge_array(cav_obs, bv_obs):
        CF_info, bv_v, bv_range_CAV, bv_rangerate_CAV = TreeSearchNADEBackgroundController.is_CF(
            cav_obs, bv_obs)
        if not CF_info:
            raise ValueError("get CF challenge in non-CF mode")
        if CF_info == "CAV_BV":
            return NADEBackgroundController._hard_brake_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)
        elif CF_info == "BV_CAV":
            # if bv_range_CAV + bv_rangerate_CAV <= 0:
            #     return np.ones((len(conf.BV_ACTIONS) - 2), dtype=float)
            # else:
            #     return np.zeros((len(conf.BV_ACTIONS) - 2), dtype=float)
            # return np.zeros((len(conf.BV_ACTIONS) - 2), dtype=float)
            return NADEBackgroundController._BV_accelerate_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)

    @staticmethod
    # @profile
    def get_cav_bv_pdf(cav_obs, bv_obs):
        cav_pdf = TreeSearchNADEBackgroundController.SURROGATE_MODEL_FUNCTION(
            cav_obs)
        _, _, bv_pdf = BASE_NDD_CONTROLLER.static_get_ndd_pdf(bv_obs)
        if utils.is_lane_change(cav_obs["Ego"]):
            cav_action_dict = {"left": 0, "right": 0, "still": 1}
            cav_pdf = [0, 1, 0]
        else:
            cav_action_dict = {
                "left": cav_pdf[0], "right": cav_pdf[2], "still": cav_pdf[1]}
        if utils.is_lane_change(bv_obs["Ego"]):
            bv_action_dict = {"left": 0, "right": 0, "still": 1}
            bv_pdf[0] = 0
            bv_pdf[1] = 0
            bv_pdf[2:] = 1.0/(len(bv_pdf[2:])) * np.ones_like(bv_pdf[2:])
            bv_pdf = bv_pdf/np.sum(bv_pdf)
        else:
            bv_action_dict = {
                "left": bv_pdf[0], "right": bv_pdf[1], "still": np.sum(bv_pdf[2:])}
        return cav_action_dict, bv_action_dict, cav_pdf, bv_pdf

    @staticmethod
    # @profile
    def _calculate_criticality(bv_obs, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None, predicted_full_traj=None):
        """calculate the criticality using tree-search based algorithm

        Args:
            CAV (vehicle): the CAV in the environment
            SM_LC_prob (list): the left, still and right turn probabiltiy

        Returns:
            array: criticality array of a specific CAV
        """
        # if utils.is_lane_change(bv_obs["Ego"]):
        #     return 0, None, None
        if not SM_LC_prob:
            CAV_left_prob, CAV_still_prob, CAV_right_prob = utils._get_Surrogate_CAV_action_probability(cav_obs=CAV)
            SM_LC_prob = [CAV_left_prob, CAV_still_prob, CAV_right_prob]
        _, _, bv_pdf = BASE_NDD_CONTROLLER.static_get_ndd_pdf(bv_obs)
        if full_obs is None:
            full_obs = TreeSearchNADEBackgroundController.cav_bv_obs_to_full_obs(bv_obs, CAV)
        # if utils.is_lane_change(bv_obs["Ego"]) and conf.experiment_config["mode"] != "risk_NDE":
        #     bv_challenge_array, updated_bv_pdf = np.zeros(TreeSearchNADEBackgroundController.ACTION_NUM), np.zeros(TreeSearchNADEBackgroundController.ACTION_NUM)
        # else:
        #     bv_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
        #         full_obs, None, None, CAV["Ego"]["veh_id"], bv_obs["Ego"]["veh_id"], 0, CAV, bv_obs, predicted_full_obs, predicted_full_traj)
        bv_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
            full_obs, None, None, CAV["Ego"]["veh_id"], bv_obs["Ego"]["veh_id"], 0, CAV, bv_obs, predicted_full_obs, predicted_full_traj)
        if utils.is_lane_change(bv_obs["Ego"]):
            bv_challenge_array[0] = bv_challenge_array[1] = bv_challenge_array[22]
        bv_criticality_array = bv_pdf*bv_challenge_array
        risk = np.sum(updated_bv_pdf*bv_challenge_array)
        bv_criticality = np.sum(bv_criticality_array)
        # self.bv_challenge_array = bv_challenge_array
        # self.bv_criticality_array = bv_criticality_array
        # normed_critical_array = bv_criticality_array/bv_criticality
        # final_array = normed_critical_array * 0.9 + 0.1 * np.array(bv_pdf)
        # weight_array = (np.array(bv_pdf) + 1e-30)/(final_array+1e-30)
        return bv_criticality, bv_criticality_array, bv_challenge_array, risk
        