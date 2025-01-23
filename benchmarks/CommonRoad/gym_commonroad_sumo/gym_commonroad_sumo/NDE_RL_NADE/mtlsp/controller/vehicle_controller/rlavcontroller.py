import torch
import torch.nn as nn
import os
import numpy as np
import gym_commonroad_sumo.NDE_RL_NADE.mtlsp.utils as utils
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.controller.vehicle_controller.controller import Controller, BaseController
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.simulator import Simulator


class DuelingNet(nn.Module):
    # for CAV
    def __init__(self, n_feature, n_hidden, n_output):
        super(DuelingNet, self).__init__()
        self.exp_item = 0  # marks the num that the expPool has stored
        self.hidden = nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.adv = nn.Linear(n_hidden, n_output)
        self.val = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """Forward function in neural networks for AV agent.

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = torch.tanh(self.hidden(x))  # activation function for hidden layer
        x = torch.tanh(self.hidden2(x))
        adv = self.adv(x)
        val = self.val(x)
        x = val + adv - adv.mean()
        return x


class RLAVController(BaseController):
    cav_observation_num = 10
    feature_size = (cav_observation_num + 1) * 3
    output_size = 33
    observed_BV_num = cav_observation_num
    path = os.path.abspath(".") + '/gym_commonroad_sumo/gym_commonroad_sumo/NDE_RL_NADE/model/100000_model_CAV_agent.pth'
    device = torch.device("cpu")
    net = DuelingNet(feature_size, 256, output_size).to(device)
    checkpoint = torch.load(path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    max_acc = 2.0
    min_acc = -4.0
    acc_res = 0.2
    ACTION_LIST = {0:"left", 1:"right"}
    for i in range(2,33):
        ACTION_LIST[i] = min_acc+acc_res*(i-2)
    LENGTH = 5
    simulation_resolution = 1
    v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high = 20, 40, 0, 115, -10, 8, -4, 2
    cav_obs_range = 120

    def __init__(self, subscription_method=Simulator.subscribe_vehicle_all_information, controllertype="AVController"):
        """Initialize an AVController object.
        """        
        super().__init__(subscription_method=subscription_method,controllertype=controllertype)

    def install(self):
        super().install()
        self.vehicle.simulator.set_vehicle_color(self.vehicle.id, self.vehicle.color_red)

    # @profile
    def step(self):
        """Decide the AV maneuver in the next step.
        """        
        super().step()
        self.action = self.decision()

    def decision(self):
        action_indicator = self.get_action_indicator(self.vehicle.observation, ndd_flag = False, safety_flag = True, CAV_flag = True)
        action_indicator_after_lane_conflict = self.lane_conflict_safety_check(self.vehicle.observation, action_indicator)
        _, _, CAV_action = self.CAV_decision(self.vehicle.observation, action_indicator_after_lane_conflict, evaluation_flag=True)
        actions = {}
        if CAV_action <= 1:
            actions["lateral"] = self.ACTION_LIST[CAV_action]
            actions["longitudinal"] = 0.0
        else:
            actions["lateral"] = "central"
            actions["longitudinal"] = self.ACTION_LIST[CAV_action]
        return actions

    def get_action_indicator(self, observation, ndd_flag = False, safety_flag = True, CAV_flag = False):
        """Get the action indicator for the CAV.

        Args:
            observation (Observation): Observation of the vehicle.
            ndd_flag (bool, optional): Check whether the vehicle follows the NDD model. Defaults to False.
            safety_flag (bool, optional): Check whether safety check is necessary. Defaults to True.
            CAV_flag (bool, optional): Check whether the vehicle is CAV. Defaults to False.

        Raises:
            ValueError: If this function is called by BV, raise error.

        Returns:
            float: Action indicator.
        """        
        if CAV_flag:
            action_shape = len(self.ACTION_LIST)
            ndd_action_indicator = np.ones(action_shape)
            if ndd_flag:
                pass
            safety_action_indicator = np.ones(action_shape)
            if safety_flag:
                lateral_action_indicator = np.array([1, 1, 1])
                lateral_result = self._check_lateral_safety(observation.information, lateral_action_indicator, CAV_flag=True)
                longi_result = BaseController._check_longitudinal_safety(observation.information, np.ones(action_shape-2), lateral_result=lateral_result, CAV_flag=True)
                safety_action_indicator[0], safety_action_indicator[1] = lateral_result[0], lateral_result[2]      
                safety_action_indicator[2:] = longi_result                
            action_indicator = ndd_action_indicator * safety_action_indicator
            action_indicator = (action_indicator > 0)
            return action_indicator

        else:
            raise ValueError("Get BV action Indicator in CAV function")

    def lane_conflict_safety_check(self, observation, action_indicator_before):
        """Check lateral safety.

        Args:
            action_indicator_before (list(bool)): List of old action indicator.

        Returns:
            list(bool): New action indicator list.
        """        
        # If there is no longitudinal actions are OK or in the middle lane, then do not block the lane change probability
        CAV_info = observation.information["Ego"]
        CAV_v, CAV_x, CAV_current_lane_id = CAV_info["velocity"], CAV_info["position"][0], CAV_info["lane_index"]
        if (not action_indicator_before[2:].any()) or (CAV_current_lane_id == 1):
            return action_indicator_before

        # If there is no lane change probability, just return
        if (CAV_current_lane_id==0 and not action_indicator_before[0]) or (CAV_current_lane_id==2 and not action_indicator_before[1]):
            return action_indicator_before

        if CAV_current_lane_id == 0: candidate_BV_lane, CAV_ban_lane_change_id = 2, 0
        elif CAV_current_lane_id == 2: candidate_BV_lane, CAV_ban_lane_change_id = 0, 1
        # candidate_BV
        bvs = observation.context
        candidate_BV_dict = {}
        for veh_id in bvs.keys():
            if bvs[veh_id][82] == candidate_BV_lane:
                candidate_BV_dict[veh_id] = bvs[veh_id]
        if len(candidate_BV_dict) == 0:
            return action_indicator_before

        r_now, rr_now, r_1_second, r_2_second = [], [], [], []
        for veh_id in candidate_BV_dict.keys():
            BV_x, BV_v = candidate_BV_dict[veh_id][66][0], candidate_BV_dict[veh_id][64]
            if BV_x >= CAV_x:
                r_now_tmp = BV_x - CAV_x - self.LENGTH
                rr_now_tmp = BV_v - CAV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp*self.simulation_resolution
                acc_BV = acc_CAV = self.acc_low
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v+acc_BV, self.v_low, self.v_high), acc_BV, time_interval=self.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v+acc_CAV, self.v_low, self.v_high), acc_CAV, time_interval=self.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                r_2_second_tmp = r_1_second_tmp + BV_dis - CAV_dis
                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)
            else:
                r_now_tmp = CAV_x - BV_x - self.LENGTH
                rr_now_tmp = CAV_v - BV_v
                r_1_second_tmp = r_now_tmp + rr_now_tmp*self.simulation_resolution
                acc_BV = self.acc_low
                acc_CAV = 0
                BV_dis = utils.cal_dis_with_start_end_speed(BV_v, np.clip(BV_v+acc_BV, self.v_low, self.v_high), acc_BV, time_interval=self.simulation_resolution, v_low=self.v_low, v_high=self.v_high)
                CAV_dis = utils.cal_dis_with_start_end_speed(CAV_v, np.clip(CAV_v+acc_CAV, self.v_low, self.v_high), acc_CAV, time_interval=self.simulation_resolution, v_low=self.v_low, v_high=self.v_high)            
                r_2_second_tmp = r_1_second_tmp + CAV_dis - BV_dis
                r_now.append(r_now_tmp)
                rr_now.append(rr_now_tmp)
                r_1_second.append(r_1_second_tmp)
                r_2_second.append(r_2_second_tmp)
        r_now, r_1_second, r_2_second = np.array(r_now), np.array(r_1_second), np.array(r_2_second)
        if (r_now <= 0).any() or (r_1_second <= 0).any() or (r_2_second <= 0).any():
            # Sample to decide whether ban the lane change
            if np.random.rand() <= 1e-4:
                return action_indicator_before
                
            else:
                action_indicator_before[CAV_ban_lane_change_id] = False
                return action_indicator_before
    
        return action_indicator_before

    def CAV_decision(self, observation, action_indicator=None, evaluation_flag=False):
        """Get the CAV's decision.

        Args:
            observation (Observation): Observation of vehicle.
            action_indicator (list(bool), optional): Action indicator. Defaults to None.
            evaluation_flag (bool, optional): Check whether it is evaluation mode. Defaults to False.

        Returns:
            list(float), list(float), int: Neural network state. Action Q-list. Action index.
        """
        # If all actions are invoid, then decelerate
        if not action_indicator.any():
            action_indicator[2] = True
        assert action_indicator.any()
        state = self._transfer_to_state_input(observation)
        state = torch.from_numpy(state).float().to(self.device)
        action_Q_full = self.net(state).detach().cpu().numpy()  # 1*5
        action_Q_full[np.array(action_indicator)==0] = -np.inf
        action_id = np.argmax(action_Q_full).item()
        return state, action_Q_full, action_id

    def _transfer_to_state_input(self, observation):
        """Transform the CAV's state and set it as the input of the neural network.

        Args:
            observation (Observation): Observation of vehicle.

        Returns:
            list(float): For the normal safety agent, the state should be BV's relative position of the CAV and not related to the distance to the exit ramp.
        """  
        # Normalize
        state_df = self._normalize_state(observation)
        # Fill missing rows
        if state_df.shape[0] < self.observed_BV_num + 1:
            fake_vehicle_row = np.array([[-1, -1, -1]])  # at the back of observation box, with minimum speed and at the top lane
            # rows = np.array(fake_vehicle_row)
            rows = -np.ones((self.observed_BV_num + 1 - state_df.shape[0], fake_vehicle_row.shape[1]))
            state_df = np.append(state_df,rows,axis=0)
        return state_df.flatten()

    def _normalize_state(self, observation):
        """Normalize the observation of the vehicle.

        Args:
            observation (class): Observation data.

        Returns:
            dataframe: Normalized observation......
        """
        CAV_id = "CAV"
        ego_info = observation.information["Ego"]
        context_info = observation.context
        # simulator = self.vehicle.simulator
        # Normalize BV
        x_position_range = self.cav_obs_range
        # side_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
        # lane_num = len(side_lanes)
        # lane_num = simulator.get_vehicle_lane_number(CAV_id)
        #! tmp change to hard code 
        lane_num = 3
        lane_range = lane_num - 1

        x0 = 0
        lane_id0 = utils.remap(ego_info["lane_index"], [0, lane_range], [1, -1])
        v0 = utils.remap(ego_info["velocity"], [self.v_low, self.v_high], [-1, 1])
        rows = np.array([[x0,lane_id0,v0]])
        bvs_array = np.array([[CAV_id,0]])
        for veh_id in context_info.keys():
            dx = context_info[veh_id][66][0]-ego_info["position"][0]
            bvs_array = np.append(bvs_array,[[veh_id,abs(dx)]],axis=0)
        sorted_bvs_array = bvs_array[bvs_array[:,1].astype(np.float).argsort()]
        for i in range(1, sorted_bvs_array.shape[0]):
            if float(sorted_bvs_array[i][1]) < self.cav_obs_range and rows.shape[0] <= self.cav_observation_num:
                veh_id = sorted_bvs_array[i][0]
                x = utils.remap(context_info[veh_id][66][0]-ego_info["position"][0], [-x_position_range, x_position_range], [-1, 1])
                lane_id = utils.remap(context_info[veh_id][82], [0, lane_range], [1, -1])
                v = utils.remap(context_info[veh_id][64], [self.v_low, self.v_high], [-1, 1])
                row = np.array([[x,lane_id,v]])
                rows = np.append(rows,row,axis=0)
        return rows