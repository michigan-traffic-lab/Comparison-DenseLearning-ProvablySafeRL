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

class SafeDriver:
    def __init__(self, planner, commonroad_env, yaml_path='./safedriver/yaml_config'):
        # store unsafe motion planner object
        self.planner = planner
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
        self.lateral_and_longitudinal_speed_dynamic = self.config['lateral_and_longitudinal_speed_dynamic']
        self.lateral_speed_vehicle_coordinate_dynamic = self.config['lateral_speed_vehicle_coordinate_dynamic']
        self.cal_speed_mode = self.config['cal_speed_mode']
        self.action_transfer_priority = self.config['action_transfer_priority']
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
        self.save_info = {'observation': [], 'ego_state': [], 'drl_obs': [], 'neuralmetric_obs': [], 'info':[], 'u':[], 'intended_u':[], 'unsafe_alpha':[], 'criticality':[], 'deltaf':[], 'longitudinal_speed': [], 'lateral_speed': [], 'lateral_speed_vehicle_coordinate': []}
        self.commonroad_env = commonroad_env

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

    def plan(self, observation, ego_state, drl_obs=None, neuralmetric_obs=None, env=None):
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
        unsafe_alpha = self.planner.plan(observation)
        action = self.planner.reachable_set_manager.factor2control(unsafe_alpha)[0]
        self.save_info['unsafe_alpha'].append(unsafe_alpha)
        self.save_info['intended_u'].append(action)
        self.info = 'Controlled by unsafe planner'
        if self.unsafe_control_implemented_step > 0 and self.unsafe_control_implemented_step < self.unsafe_planner_step_num:
            action = self.last_action
        else:
            self.unsafe_control_implemented_step = 0
        self.unsafe_control_implemented_step += 1

        # replan if criticality is high
        if drl_obs is not None and neuralmetric_obs is not None:
            drl_obs[0] = self.longitudinal_speed_remap # the longitudinal speed in SUMO is actually the longitudinal speed in vehicle coordinate, but we need the x-axis speed in road coordinate
            drl_obs[1] = self.lateral_speed_remap # update the lateral speed from safedriver, which is 0 in the drl_obs
            criticality = self.get_criticality(neuralmetric_obs)
            if criticality >= self.criticality_thresh:
                safe_action = self.safedriver_model.compute_action(drl_obs, True, False)
                safe_action[1] = safe_action[1].clip(-10, 10)
                self.info = 'Controlled by safe driver, safe action: \n acceleration={:.5f}, deltaf={:.5f}'.format(safe_action[0], safe_action[1]/180*math.pi)
                action = self.safe_action_to_control(safe_action)
                self.unsafe_control_implemented_step = 0
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

        else:
            criticality = 0
        try:
            self.commonroad_env.sumo_cr_simulator_local.simulation.sumo_sim.sim.env.vehicle_list["CAV"].controller.store_criticality(criticality)
        except:
            pass

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
        self.save_info['drl_obs'].append(drl_obs)
        self.save_info['neuralmetric_obs'].append(neuralmetric_obs)
        self.save_info['info'].append(self.info)
        self.save_info['u'].append(action)
        self.save_info['criticality'].append(criticality)
        self.save_info['deltaf'].append(deltaf)
        self.save_info['lateral_speed'].append(self.lateral_speed)
        self.save_info['longitudinal_speed'].append(self.longitudinal_speed)
        self.save_info['lateral_speed_vehicle_coordinate'].append(self.lateral_speed_vehicle_coordinate)

        return action

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
            new_states = self.get_new_state_simple(new_states, acc, yaw_rate)

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
        new_states = self.get_new_state_simple(states, action[0], action[1])

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
        new_states = self.get_new_state_simple(states, action[0], action[1])
        dx = new_states[1] - states[1]
        current_v = self.longitudinal_speed
        new_v = dx / self.time_step * 2 - current_v
        acc = (new_v - current_v) / self.time_step
        dy = new_states[2] - states[2]
        new_lateral_v = dy / self.time_step * 2 - self.lateral_speed
        acc_lat = (new_lateral_v - self.lateral_speed) / self.time_step
        return [acc_x - acc, acc_y - acc_lat]
    
    def get_steering_angle(self, action):
        # convert yaw rate to steering angle
        Caf, Car = self.parameters["Caf"], self.parameters["Car"]
        a, b, Iz = self.parameters["a"], self.parameters["L"]-self.parameters["a"], self.parameters["Iz"]
        u = self.ego_states_list[0] # longitudinal speed in vehicle coordinate
        v = self.lateral_speed_vehicle_coordinate # lateral speed in vehicle coordinate
        r = self.ego_states_list[5] # yaw rate in absolute coordinate
        dr = (action[1] - r) / self.time_step
        deltaf = (-dr - (b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*(-r))*Iz/(a*Caf)
        return deltaf

    def get_steering_angle_accurate(self, action):
        new_states = self.get_new_state_simple(self.ego_states_list, action[0], action[1])
        deltaf0 = self.get_steering_angle(action)
        deltaf = fsolve(
            self.solve_for_control_inputs_deltaf,
            np.array([deltaf0]),
            #args=(np.array([new_states[1],new_states[2]]),),
            args=(np.array([new_states[3]]),),
        )  # solve for the control inputs that approximate the x, y position
        
        return deltaf[0]

    def update_ego_speed(self, ego_state):
        # update yaw rate for complex dynamic, lateral speed, longitudinal speed and lateral speed in vehicle coordinate of ego vehicle
        states = self.ego_states_list
        new_states = self.transfer_ego_state_to_list(ego_state)
        new_states_complex = self.get_new_state_complex(states, self.last_action[0], self.last_deltaf)
        self.yaw_rate_complex = new_states_complex[5]
        self.phi_complex = new_states_complex[3]

        if self.lateral_and_longitudinal_speed_dynamic == 'real':
            dy = new_states[2] - states[2]
            self.lateral_speed = self.calculate_avg_speed(dy, self.lateral_speed)
            dx = new_states[1] - states[1]
            self.longitudinal_speed = self.calculate_avg_speed(dx, self.longitudinal_speed)
        elif self.lateral_and_longitudinal_speed_dynamic == 'complex':
            dy = new_states_complex[2] - states[2]
            self.lateral_speed = self.calculate_avg_speed(dy, self.lateral_speed)
            dx = new_states_complex[1] - states[1]
            self.longitudinal_speed = self.calculate_avg_speed(dx, self.longitudinal_speed)
        else:
            print('unknown lateral and longitudinal speed dynamic')
            self.longitudinal_speed = new_states[0]
            self.lateral_speed = 0
        self.lateral_speed_remap = mtlsp_utils.remap(self.lateral_speed,
                    [-conf.lat_v_max, conf.lat_v_max], [-1, 1])
        self.longitudinal_speed_remap = min(mtlsp_utils.remap(self.longitudinal_speed, [0, conf.v_max], [-1, 1]), 1)
        if self.lateral_speed_vehicle_coordinate_dynamic == 'real':
            self.lateral_speed_vehicle_coordinate = self.longitudinal_speed * math.sin(new_states[3]) - self.lateral_speed * math.cos(new_states[3])
        elif self.lateral_speed_vehicle_coordinate_dynamic == 'complex':
            self.lateral_speed_vehicle_coordinate = new_states_complex[4]
        else:
            print('unknown lateral speed vehicle coordinate dynamic')
            self.lateral_speed_vehicle_coordinate = 0

    def calculate_avg_speed(self, dx, last_speed):
        if self.cal_speed_mode == 'current': # calculate the speed at the current time step
            return dx / self.time_step
        elif self.cal_speed_mode == 'next': # calculate the speed at the next time step
            return 2 * dx / self.time_step - last_speed
        else:
            print('unknown speed calculation mode')
            return dx / self.time_step

    def safe_action_to_control(self, safe_action):
        # convert safe action to control
        acc = safe_action[0] # longitudinal acceleration in vehicle coordinate 
        deltaf = safe_action[1]/180*math.pi # steering angle in vehicle coordinate
        states = self.ego_states_list
        #states[3] = self.phi_complex
        new_states = self.get_new_state_complex(states, acc, deltaf)
        if self.action_transfer_priority == 'yaw_rate': # fit yaw_rate first
            action = np.array([acc, new_states[5]])
        elif self.action_transfer_priority == 'position':
            action = fsolve(
                self.solve_for_control_inputs_position,
                [float(acc), float(new_states[5])],
                args=(np.array([new_states[1],new_states[2]]),),
            )  # solve for the control inputs that approximate the x, y position
        elif self.action_transfer_priority == 'simple':
            action = np.array([acc, -deltaf*states[0]/self.parameters["L"]])
        else:
            print('unknown action transfer priority')
            action = np.array([acc, new_states[5]])
        if action[1] == self.min_yaw_rate or action[1] == self.max_yaw_rate:
            self.info += ', clipped'
        action[0] = action[0].clip(self.min_acc, self.max_acc)
        action[1] = action[1].clip(self.min_yaw_rate, self.max_yaw_rate)
        return action

    def solve_for_control_inputs_position(self, action, position_dict):
        action[0] = action[0].clip(self.min_acc, self.max_acc)
        action[1] = action[1].clip(self.min_yaw_rate, self.max_yaw_rate)
        x, y = position_dict[0], position_dict[1]
        states = self.ego_states_list
        new_states = self.get_new_state_simple(states, action[0], action[1])
        new_x = new_states[1]
        new_y = new_states[2]
        return [x - new_x, y - new_y]
    
    '''
    def solve_for_control_inputs_deltaf(self, deltaf, info):
        deltaf[0] = deltaf[0].clip(-10/180*math.pi, 10/180*math.pi)
        states = self.ego_states_list
        acc = self.last_action[0]
        new_states = self.get_new_state_complex(states, acc, deltaf[0])
        x,y = info[0], info[1]
        new_x = new_states[1]
        new_y = new_states[2]
        return (x - new_x)**2+ (y - new_y)**2
    '''

    def solve_for_control_inputs_deltaf(self, deltaf, info):
        deltaf[0] = deltaf[0].clip(-10/180*math.pi, 10/180*math.pi)
        states = self.ego_states_list
        acc = self.last_action[0]
        new_states = self.get_new_state_complex(states, acc, deltaf[0])
        phi = info[0]
        new_phi = new_states[3]
        return (phi-new_phi)**2

    def transfer_ego_state_to_list(self, ego_state):
        u = ego_state.velocity # longitudinal speed in vehicle coordinate
        x = ego_state.position[0] # longitudinal position in road coordinate
        y = ego_state.position[1] # lateral position in road coordinate
        phi = ego_state.orientation # heading angle in road coordinate
        v = self.lateral_speed_vehicle_coordinate # lateral speed in vehicle coordinate
        r = self.yaw_rate_complex # yaw rate in road coordinate
        states = [u, x, y, phi, v, r]
        return states

    def get_new_state_simple(self, states, acc, yaw_rate):
        # PMNonlinear Model
        new_states = copy.deepcopy(states)
        iteration = 100
        for _ in range(iteration):
            new_states[1] += new_states[0] * np.cos(new_states[3]) * self.time_step / iteration
            new_states[2] += new_states[0] * np.sin(new_states[3]) * self.time_step / iteration
            new_states[0] += acc * self.time_step / iteration
            new_states[3] += yaw_rate * self.time_step / iteration
        return new_states

    def get_new_state_complex(self, states, acc, deltaf):
        # fourth order Runge-Kutta method
        k1 = self.state_update_complex(states, acc, deltaf)
        states_k2 = [states[i]+self.time_step*k1[i]/2 for i in range(len(states))]
        k2 = self.state_update_complex(states_k2, acc, deltaf)
        states_k3 = [states[i]+self.time_step*k2[i]/2 for i in range(len(states))]
        k3 = self.state_update_complex(states_k3, acc, deltaf)
        states_k4 = [states[i]+self.time_step*k3[i] for i in range(len(states))]
        k4 = self.state_update_complex(states_k4, acc, deltaf)
        new_states = [states[i]+self.time_step*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(len(states))]

        return new_states

    def state_update_complex(self, states, acc, deltaf):
        # acc: longitudinal acceleration in vehicle coordinate 
        # deltaf: steering angle in vehicle coordinate
        u = states[0] # longitudinal speed in vehicle coordinate
        x = states[1] # longitudinal position in road coordinate
        y = states[2] # lateral position in road coordinate
        phi = states[3] # vehicle heading in road coordinate
        v = states[4] # lateral speed in vehicle coordinate
        r = states[5] # yaw rate in road coordinate
        Caf, Car = self.parameters["Caf"], self.parameters["Car"]
        a, b, m, Iz = self.parameters["a"], self.parameters["L"]-self.parameters["a"], self.parameters['m'], self.parameters["Iz"]
        # Use Runge–Kutta method
        k = [
            acc, 
            u*math.cos(phi)+v*math.sin(phi), 
            -v*math.cos(phi)+u*math.sin(phi),
            r, 
            -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*(-r)+(Caf/m)*deltaf, 
            -((b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*(-r)+a*(Caf/Iz)*deltaf) # change dr to -dr because the meaning of yaw_rate in Commonroad (positive when turning left) is opposite to SUMO (positive when turning right), and deltaf is positive when turning right
        ]
        return k
    

    def original_update_vehicle_real_states(self, original_states, action, parameters, duration):
        """Get the next vehicle states based on simple vehicle dynamic model (bicycle model).

        Args:
            original_states (list): Vehicle states including longitudinal speed in vehicle coordinate, longitudinal position in road coordinate, lateral position in road coordinate, heading in absolute coordinate, lateral speed in vehicle coordinate, yaw rate in absolute coordinate.
            action (dict): Next action including longitudinal acceleration in vehicle coordinate and steering angle.
            parameters (dict): Vehicle dynamics parameters including a, L, m, Iz, Caf, Car.
            duration (float): Simulation time step.

        Returns:
            list: New vehicle states with the same format as the original states. 
        """    
        # first assume straight road
        au = action["acceleration"] # longitudinal acceleration in vehicle coordinate 
        deltaf = action["steering_angle"]/180*math.pi # steering angle in vehicle coordinate
        u = original_states[0] # longitudinal speed in vehicle coordinate
        x = original_states[1] # longitudinal position in road coordinate
        y = original_states[2] # lateral position in road coordinate
        original_states[3] = math.pi/2
        phi = original_states[3] # vehicle heading in absolute coordinate
        phid = math.pi/2
        v = original_states[4] # lateral speed in vehicle coordinate
        r = original_states[5] # yaw rate in absolute coordinate
        whole_states = [original_states]
        Caf, Car = parameters["Caf"], parameters["Car"]
        a, b, m, Iz = parameters["a"], parameters["L"]-parameters["a"], parameters["m"], parameters["Iz"]
        # Use Runge–Kutta method
        k1 = self.original_helper_state_update(original_states, action, parameters)
        states_k2 = [original_states[i]+duration*k1[i]/2 for i in range(len(original_states))]
        k2 = self.original_helper_state_update(states_k2, action, parameters)
        states_k3 = [original_states[i]+duration*k2[i]/2 for i in range(len(original_states))]
        k3 = self.original_helper_state_update(states_k3, action, parameters)
        states_k4 = [original_states[i]+duration*k3[i] for i in range(len(original_states))]
        k4 = self.original_helper_state_update(states_k4, action, parameters)
        RK_states = [original_states[i]+duration*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6 for i in range(len(original_states))]
        # Euler method
        # dt = 0.0001
        # num_step = int(duration/dt)
        # remaining_time = duration - num_step*dt
        # for step in range(num_step):
        #     dudt = au
        #     dxdt = u*math.cos(phi-phid)-v*math.sin(phi-phid)
        #     dydt = v*math.cos(phi-phid)+u*math.sin(phi-phid)
        #     dphidt = r
        #     dvdt = -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+(Caf/m)*deltaf
        #     drdt = (b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*r+a*(Caf/Iz)*deltaf
        #     states = [u+dudt*dt, x+dxdt*dt, y-dydt*dt, phi+dphidt*dt, v+dvdt*dt, r+drdt*dt]
        #     whole_states.append(states)
        #     u,x,y,phi,v,r = states
        # if remaining_time > 0:
        #     dudt = au
        #     dxdt = u*math.cos(phi)-v*math.sin(phi)
        #     dydt = v*math.cos(phi)+u*math.sin(phi)
        #     dphidt = r
        #     dvdt = -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+Caf/m*deltaf
        #     drdt = (b*Car-a*Caf)/(Iz*u)*v-(a^2*Caf+b^2*Car)/(Iz*u)*r+a*Caf/Iz*deltaf
        #     states = [u+dudt*remaining_time, x+dxdt*remaining_time, y-dydt*remaining_time, phi+dphidt*remaining_time, v+dvdt*remaining_time, r+drdt*remaining_time]
        #     whole_states.append(states)
        #     u,x,y,phi,v,r = states
        RK_states[3] = math.pi/2-RK_states[3]
        RK_states[5] = -RK_states[5]
        return RK_states
        

    def original_helper_state_update(self, states, action, parameters):
        au = action["acceleration"] # longitudinal acceleration in vehicle coordinate 
        deltaf = action["steering_angle"]/180*math.pi # steering angle in vehicle coordinate, clockwise is positive
        u = states[0] # longitudinal speed in vehicle coordinate
        x = states[1] # longitudinal position in road coordinate
        y = states[2] # lateral position in road coordinate
        phi = states[3] # vehicle heading in absolute coordinate, clockwise is positive, north is 0, east is pi/2
        phid = math.pi/2
        v = states[4] # lateral speed in vehicle coordinate, right is positive
        r = states[5] # yaw rate in absolute coordinate, clockwise is positive
        Caf, Car = parameters["Caf"], parameters["Car"]
        a, b, m, Iz = parameters["a"], parameters["L"]-parameters["a"], parameters["m"], parameters["Iz"]
        # Use Runge–Kutta method
        k = [
            au, 
            u*math.cos(phi-phid)-v*math.sin(phi-phid), 
            -(v*math.cos(phi-phid)+u*math.sin(phi-phid)),
            r, 
            -(Caf+Car)/(m*u)*v+((b*Car-a*Caf)/(m*u)-u)*r+(Caf/m)*deltaf, 
            (b*Car-a*Caf)/(Iz*u)*v-((a**2)*Caf+(b**2)*Car)/(Iz*u)*r+a*(Caf/Iz)*deltaf
        ]
        return k