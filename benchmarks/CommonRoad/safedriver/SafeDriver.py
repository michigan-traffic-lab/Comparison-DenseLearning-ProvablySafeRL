from gym_commonroad_sumo.NDE_RL_NADE.controller.neuralmetric import NN_Metric
import numpy as np
import os
import yaml
import math
from gym_commonroad_sumo.NDE_RL_NADE.conf import conf
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp import utils as mtlsp_utils
import copy
from gym_commonroad_sumo.NDE_RL_NADE.controller.RSS_model import highway_RSS_model
from scipy.optimize import fsolve
from scipy.integrate import odeint
import time

class SafeDriver:
    def __init__(self, planner, base, yaml_path='./safedriver/yaml_config'):
        # store unsafe motion planner object
        self.planner = planner
        self.base_mode = base
        self.max_yaw_rate = self.planner.reachable_set_manager.factor2control(np.array([1, 1, 1, 1]))[0][1]
        self.min_yaw_rate = self.planner.reachable_set_manager.factor2control(np.array([-1, -1, -1, -1]))[0][1]
        self.max_acc = self.planner.reachable_set_manager.factor2control(np.array([1, 1, 1, 1]))[0][0]
        self.min_acc = self.planner.reachable_set_manager.factor2control(np.array([-1, -1, -1, -1]))[0][0]
        self.road_bound = [52, 40] # road boundary on y-axis

        # get the neural metric
        self.nn_metric_yaml_path = os.path.join(yaml_path, 'nn_metric.yaml')
        self.nn_metric = NN_Metric(self.nn_metric_yaml_path)

        # get safedriver config
        try:
            with open(os.path.join(yaml_path, 'safedriver.yaml'), 'r') as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            print("Yaml configuration file not successfully loaded:", e)

        self.safedriver_model = conf.load_ray_agent(self.config['pytorch_model_path_list'])
        self.criticality_thresh = self.config['criticality_thresh']
        self.RSS_flag = self.config['RSS_flag']
        self.cal_speed_mode = self.config['cal_speed_mode']
        self.relax_yaw_rate_clip = self.config['relax_yaw_rate_clip']
        if self.relax_yaw_rate_clip:
            self.max_yaw_rate = max(0.4, self.max_yaw_rate)
            self.min_yaw_rate = min(-0.4, self.min_yaw_rate)

        self.parameters = {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500, # mass (kg)
            "Iz": 2420, # yaw moment of inertia (kg-m^2)
            "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
        }
        self.time_step = 0.1
        self.longitudinal_speed = 0
        self.longitudinal_speed_remap = 0
        self.lateral_speed = 0
        self.lateral_speed_vehicle_coordinate = 0
        self.lateral_speed_remap = 0
        self.yaw_rate_complex = 0
        self.phi_complex = 0
        self.ego_states_list = []
        self.last_action = None
        self.last_deltaf = 0
        self.unsafe_control_implemented_step = 0
        self.critical_flag = False
        self.unsafe_planner_dt = 0.4
        self.unsafe_planner_step_num = 4
        self.RSS_model = highway_RSS_model()
        self.info = None
        self.save_info = {'observation': [], 'ego_state': [], 'drl_obs': [], 'neuralmetric_obs': [], 'info':[], 'u':[], 'intended_u':[], 'unsafe_alpha':[], 'criticality':[], 'deltaf':[], 'longitudinal_speed': [], 'lateral_speed': [], 'lateral_speed_vehicle_coordinate': [], 'safe_action': [], 'ego_state_list_sumo': []}

    def set_time_step(self, dt, unsafe_planner_dt=0.4):
        self.time_step = dt
        self.unsafe_planner_dt = unsafe_planner_dt
        self.unsafe_planner_step_num = int(self.unsafe_planner_dt / self.time_step)

    def reset(self):
        self.longitudinal_speed = 0
        self.longitudinal_speed_remap = 0
        self.lateral_speed = 0
        self.lateral_speed_vehicle_coordinate = 0
        self.lateral_speed_remap = 0
        self.yaw_rate_complex = 0
        self.phi_complex = 0
        self.last_action = None
        self.last_deltaf = 0
        self.unsafe_control_implemented_step = 0
        self.critical_flag = False
        self.RSS_model = highway_RSS_model()
        self.save_info = {'observation': [], 'ego_state': [], 'drl_obs': [], 'neuralmetric_obs': [], 'info':[], 'u':[], 'intended_u':[], 'unsafe_alpha':[], 'criticality':[], 'deltaf':[], 'longitudinal_speed': [], 'lateral_speed': [], 'lateral_speed_vehicle_coordinate': [], 'safe_action': [], 'ego_state_list_sumo': []}

    def plan(self, observation, ego_state, drl_obs=None, neuralmetric_obs=None, env=None, scenario=None):
        # obtain control command from the reinforcement learning agent
        if self.last_action is not None:
            self.update_ego_speed(ego_state)
        else:
            self.longitudinal_speed = ego_state.velocity
            self.longitudinal_speed_remap = min(
                mtlsp_utils.remap(self.longitudinal_speed, [0, conf.v_max], [-1, 1]), 1
            )
            self.lateral_speed = 0
            self.lateral_speed_remap = 0
            self.lateral_speed_vehicle_coordinate = 0
            self.yaw_rate_complex = 0
            self.phi_complex = 0

        self.ego_states_list = self.transfer_ego_state_to_list(ego_state)
        time_unsafe = 0
        if self.unsafe_control_implemented_step > 0 and self.unsafe_control_implemented_step < self.unsafe_planner_step_num:
            action = self.last_action
            unsafe_alpha = self.save_info['unsafe_alpha'][-1]
        else:
            start_time1 = time.time()
            self.unsafe_control_implemented_step = 0
            if self.base_mode == 'safe':
                action, intended_u, opt_time = self.planner.plan(observation, scenario)
                unsafe_alpha = self.planner.save_info['unsafe_alpha'][-1]
            else:
                unsafe_alpha = self.planner.plan(observation)
                action = self.planner.reachable_set_manager.factor2control(unsafe_alpha)[0]
            time_unsafe = time.time() - start_time1
        self.save_info['unsafe_alpha'].append(unsafe_alpha)
        self.save_info['intended_u'].append(action)
        self.info = 'Controlled by unsafe planner'
        self.unsafe_control_implemented_step += 1

        # replan if criticality is high
        time_safedriver = 0
        if drl_obs is not None and neuralmetric_obs is not None:
            start_time2 = time.time()
            drl_obs[0] = self.longitudinal_speed_remap # the longitudinal speed in SUMO is actually the longitudinal speed in vehicle coordinate, but we need the x-axis speed in road coordinate
            drl_obs[1] = self.lateral_speed_remap # update the lateral speed from safedriver, which is 0 in the drl_obs
            criticality = self.get_criticality(neuralmetric_obs)
            if criticality >= self.criticality_thresh:
                safe_action = self.safedriver_model.compute_action(drl_obs, True, False)
                self.save_info['safe_action'].append(safe_action)
                self.info = 'Controlled by safe driver, safe action: \n acceleration={:.5f}, deltaf={:.5f}'.format(safe_action[0], safe_action[1]/180*math.pi)
                action = self.safe_action_to_control(safe_action)
                self.unsafe_control_implemented_step = 0
            elif self.base_mode == 'test':
                safe_action = self.safedriver_model.compute_action(drl_obs, True, False)
                self.save_info['safe_action'].append(safe_action)
                self.safe_action_to_control(safe_action)
            else:
                self.save_info['safe_action'].append([None,None])
            if env is not None and self.RSS_flag:
                self.RSS_check(action, env)
                if (
                    criticality >= self.criticality_thresh
                    and self.RSS_model.cav_veh.RSS_control
                ):
                    action = fsolve(
                        self.solve_for_control_inputs,
                        [float(action[0]), float(action[1])],
                        args=(self.RSS_model.cav_veh.action,),
                    )  # solve for the control inputs that approximate the RSS control
                    action[0] = action[0].clip(self.min_acc, self.max_acc)
                    action[1] = action[1].clip(self.min_yaw_rate, self.max_yaw_rate)
                    self.info += "\n Use RSS, corrected action: \n acceleration={:.5f}, yaw rate={:.5f}".format(action[0], action[1])
            time_safedriver = time.time() - start_time2
        else:
            criticality = 0
            self.save_info['safe_action'].append([None,None])

        self.last_action = action
        action = self.off_road_check(action)
        if self.last_action[1] != action[1]:
            self.last_action = action
            self.unsafe_control_implemented_step = 0
        deltaf = self.get_steering_angle_accurate(action)
        self.last_deltaf = deltaf

        self.info += '\n Step info: criticality={:.5f} \n lateral speed (road)={:.5f} lateral speed (vehicle)={:.5f}'.format(criticality, self.lateral_speed, self.lateral_speed_vehicle_coordinate)
        self.info += '\n Yaw rate: expected={:.5f}, real={:.5f}'.format(self.yaw_rate_complex, ego_state.yaw_rate)
        self.info += '\n Orientation: expected={:.5f}, real={:.5f}'.format(self.phi_complex, ego_state.orientation)
        self.info += '\n Final deltaf={:.5f}'.format(deltaf)

        # reset to real states
        self.yaw_rate_complex = ego_state.yaw_rate
        self.phi_complex = ego_state.orientation
        self.lateral_speed_vehicle_coordinate = self.longitudinal_speed * math.sin(ego_state.orientation) - self.lateral_speed * math.cos(ego_state.orientation)

        # update save info
        self.save_info['observation'].append(observation)
        self.save_info['ego_state'].append(ego_state)
        if drl_obs is not None:
            self.save_info['drl_obs'].append(drl_obs.tolist())
            self.save_info['neuralmetric_obs'].append(self.nn_metric.normalize(copy.deepcopy(neuralmetric_obs))[0].tolist())
        else:
            self.save_info['drl_obs'].append(None)
            self.save_info['neuralmetric_obs'].append(None)
        self.save_info['info'].append(self.info)
        self.save_info['u'].append(action)
        self.save_info['criticality'].append(criticality)
        self.save_info['deltaf'].append(deltaf)
        self.save_info['lateral_speed'].append(self.lateral_speed)
        self.save_info['longitudinal_speed'].append(self.longitudinal_speed)
        self.save_info['lateral_speed_vehicle_coordinate'].append(self.lateral_speed_vehicle_coordinate)
        veh_state = copy.deepcopy(self.ego_states_list)
        self.save_info['ego_state_list_sumo'].append(veh_state)

        return action, time_unsafe, time_safedriver

    def get_criticality(self, neuralmetric_obs):
        # get criticality from neural metric
        criticality = self.nn_metric.inference(self.nn_metric.normalize(copy.deepcopy(neuralmetric_obs)))
        return criticality

    def off_road_check(self, action):
        # check if the vehicle is nearly off road and replan if true
        acc = action[0]
        yaw_rate = action[1]
        new_action = action
        new_states = self.ego_states_list
        # get the state after 3 time steps if maintain the current action
        for _ in range(3):
            new_states = self.get_new_states_cr(new_states, acc, yaw_rate)

        if new_states[2] + 5 / 2 * math.sin(new_states[3]) + 1.8 / 2 * math.cos(new_states[3]) > self.road_bound[0] - 1:
            if self.ego_states_list[3] + self.min_yaw_rate * self.time_step < 0:
                new_action[1] = min(self.min_yaw_rate / 2, action[1])
            else:
                new_action[1] = self.min_yaw_rate
            self.info += '\n Off road check correction'
        elif new_states[2] + 5 / 2 * math.sin(new_states[3]) - 1.8 / 2 * math.cos(new_states[3]) < self.road_bound[1] + 1:
            if self.ego_states_list[3] + self.max_yaw_rate * self.time_step > 0:
                new_action[1] = max(self.max_yaw_rate / 2, action[1])
            else:
                new_action[1] = self.max_yaw_rate
            self.info += '\n Off road check correction'

        return new_action

    def RSS_check(self, action, env):
        # use RSS to check if the action is safe
        states = self.ego_states_list
        new_states = self.get_new_states_cr(states, action[0], action[1])

        dx = new_states[1] - states[1]
        current_v = self.longitudinal_speed
        new_v = dx / self.time_step * 2 - current_v
        acc = (new_v - current_v) / self.time_step
        dy = new_states[2] - states[2]
        new_lateral_v = dy / self.time_step * 2 - self.lateral_speed
        acc_lat = (new_lateral_v - self.lateral_speed) / self.time_step
        RSS_restriction = self.RSS_model.RSS_act_CAV(env, {"acc_x": acc, "acc_y": acc_lat}, 
                                                     longitudinal_speed = self.longitudinal_speed,
                                                     lateral_speed=self.lateral_speed)
        RSS_action = self.RSS_model.RSS_step_CAV(env=env) # return a_x and a_y in road coordinate

    def solve_for_control_inputs(self, action, acc_dict):
        acc_x, acc_y = acc_dict["acc_x"], acc_dict["acc_y"]
        states = self.ego_states_list
        new_states = self.get_new_states_cr(states, action[0], action[1])
        dx = new_states[1] - states[1]
        current_v = self.longitudinal_speed
        new_v = dx / self.time_step * 2 - current_v
        acc = (new_v - current_v) / self.time_step
        dy = new_states[2] - states[2]
        new_lateral_v = dy / self.time_step * 2 - self.lateral_speed
        acc_lat = (new_lateral_v - self.lateral_speed) / self.time_step
        return [acc_x - acc, acc_y - acc_lat]

    def get_steering_angle_accurate(self, action):
        # get the accurate steering angle from yaw rate
        state = self.ego_states_list
        deltaf = -action[1]/state[0]*self.parameters["L"]
        
        return deltaf

    def update_ego_speed(self, ego_state):
        # update yaw rate for complex dynamic, lateral speed, longitudinal speed and lateral speed in vehicle coordinate of ego vehicle
        states = self.ego_states_list
        new_states = self.transfer_ego_state_to_list(ego_state)
        self.yaw_rate_complex = new_states[5]
        self.phi_complex = new_states[3]

        dy = new_states[2] - states[2]
        self.lateral_speed = self.calculate_avg_speed(dy, self.lateral_speed)
        dx = new_states[1] - states[1]
        self.longitudinal_speed = self.calculate_avg_speed(dx, self.longitudinal_speed)
        self.lateral_speed_remap = mtlsp_utils.remap(self.lateral_speed,
                    [-conf.lat_v_max, conf.lat_v_max], [-1, 1])
        self.longitudinal_speed_remap = min(mtlsp_utils.remap(self.longitudinal_speed, [0, conf.v_max], [-1, 1]), 1)
        self.lateral_speed_vehicle_coordinate = self.longitudinal_speed * math.sin(new_states[3]) - self.lateral_speed * math.cos(new_states[3])

    def calculate_avg_speed(self, dx, last_speed):
        if self.cal_speed_mode == 'current': # calculate the speed at the current time step
            return dx / self.time_step
        elif self.cal_speed_mode == 'next': # calculate the speed at the next time step
            return 2 * dx / self.time_step - last_speed
        else:
            print('unknown speed calculation mode')
            return 2 * dx / self.time_step - last_speed

    def safe_action_to_control(self, safe_action):
        # convert safe action to control
        safe_action[1] = safe_action[1].clip(-10, 10)
        acc = safe_action[0] # longitudinal acceleration in vehicle coordinate 
        deltaf = safe_action[1]/180*math.pi # steering angle in vehicle coordinate
        states = self.ego_states_list
        action = np.array([acc, -deltaf*states[0]/self.parameters["L"]])
        action[0] = action[0].clip(self.min_acc, self.max_acc)
        action[1] = action[1].clip(self.min_yaw_rate, self.max_yaw_rate)
        if action[1] == self.min_yaw_rate or action[1] == self.max_yaw_rate:
            self.info += ', clipped'
        return action

    def transfer_ego_state_to_list(self, ego_state):
        u = ego_state.velocity # longitudinal speed in vehicle coordinate
        x = ego_state.position[0] # longitudinal position in road coordinate
        y = ego_state.position[1] # lateral position in road coordinate
        phi = ego_state.orientation # heading angle in road coordinate
        v = self.lateral_speed_vehicle_coordinate # lateral speed in vehicle coordinate
        r = ego_state.yaw_rate # yaw rate in road coordinate
        states = [u, x, y, phi, v, r]
        return states

    def get_new_states_simple(self, states, acc, yaw_rate):
        # PMNonlinear Model
        new_states = copy.deepcopy(states)
        iteration = 100
        for _ in range(iteration):
            new_states[1] += new_states[0] * np.cos(new_states[3]) * self.time_step / iteration
            new_states[2] += new_states[0] * np.sin(new_states[3]) * self.time_step / iteration
            new_states[0] += acc * self.time_step / iteration
            new_states[3] += yaw_rate * self.time_step / iteration
        return new_states
    
    @staticmethod
    def dynamics(t, x, u):
        """
        Point Mass model dynamics function. Overrides the dynamics function of VehicleDynamics for PointMass model.

        :param t:
        :param x: state values, [position x, position y, velocity, orientation]
        :param u: input values, [acceleration, yaw rate]

        :return:
        """
        return [
            x[2] * np.cos(x[3]),
            x[2] * np.sin(x[3]),
            u[0],
            u[1]
        ]

    def get_new_states_cr(self, states, acc, yaw_rate):
        values = [
            states[1],
            states[2],
            states[0],
            states[3]
        ]
        x_current = np.array(values)
        u_input = np.array([acc, yaw_rate])
        _, x1 = odeint(self.dynamics, x_current, [0.0, self.time_step], args=(u_input,), tfirst=True)
        return [
            x1[2],x1[0],x1[1],x1[3],states[4],yaw_rate
        ]