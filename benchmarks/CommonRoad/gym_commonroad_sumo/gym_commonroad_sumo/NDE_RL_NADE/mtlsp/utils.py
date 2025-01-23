import uuid
import numpy as np
import math
import torch
import torch.nn as nn
import bisect
import xml.etree.ElementTree as ET
import json


def generate_unique_bv_id():
    """Randomly generate an ID of the background vehicle

    Returns:
        str: ID of the background vehicle
    """
    return 'BV_'+str(uuid.uuid4())

def remap(v, x, y): 
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])

def check_equal(x, y, error):
    """Check if x is approximately equal to y considering the given error.

    Args:
        x (float): Parameter 1.
        y (float): Parameter 2.
        error (float): Specified error.

    Returns:
        bool: True is x and y are close enough. Otherwise, False.
    """
    if abs(x-y) <= error:
        return True
    else:
        return False

def cal_dis_with_start_end_speed(v_start, v_end, acc, time_interval=1.0, v_low=20, v_high=40):
    """Calculate the travel distance with start and end speed and acceleration.

    Args:
        v_start (float): Start speed [m/s].
        v_end (float): End speed [m/s].
        acc (float): Acceleration [m/s^2].
        time_interval (float, optional): Time interval [s]. Defaults to 1.0.

    Returns:
        float: Travel distance in the time interval.
    """
    if v_end == v_low or v_end == v_high:
        t_1 = (v_end-v_start)/acc if acc != 0 else 0
        t_2 = time_interval - t_1
        dis = v_start*t_1 + 0.5*(acc)*(t_1**2) + v_end*t_2
    else:
        dis = ((v_start+v_end)/2)*time_interval
    return dis

def cal_euclidean_dist(veh1_position=None, veh2_position=None):
    """Calculate Euclidean distance between two vehicles.

    Args:
        veh1_position (tuple, optional): Position of Vehicle 1 [m]. Defaults to None.
        veh2_position (tuple, optional): Position of Vehicle 2 [m]. Defaults to None.

    Raises:
        ValueError: If the position of fewer than two vehicles are provided, raise error.

    Returns:
        float: Euclidean distance between two vehicles [m].
    """
    if veh1_position is None or veh2_position is None:
        raise ValueError("Fewer than two vehicles are provided!")
    veh1_x, veh1_y = veh1_position[0], veh1_position[1]
    veh2_x, veh2_y = veh2_position[0], veh2_position[1]
    return math.sqrt(pow(veh1_x-veh2_x, 2)+pow(veh1_y-veh2_y, 2))

def load_trajs_from_fcdfile(file_path):
    vehicle_trajs = {}
    tree = ET.parse(file_path)
    root = tree.getroot()
    for time_step in root.getchildren():
        time = float(time_step.attrib["time"])
        vehicles_dict = {}
        for veh in time_step:
            veh_info = veh.attrib
            veh_id = veh_info["id"]
            vehicles_dict[veh_id] = veh_info
        vehicle_trajs[str(round(float(time),1))] = vehicles_dict
    return vehicle_trajs

def load_trajs_from_jsonfile(file_path):
    vehicle_trajs = {}
    # file_path = file_path_info[1]+"/"+str(file_path_info[0][1])+".fcd.json"
    with open(file_path) as fo:
        fcd_info = json.load(fo)
        for time_step_info in fcd_info["fcd-export"]["timestep"]:
            time = time_step_info["@time"]
            vehicles_dict = {}
            for veh in time_step_info["vehicle"]:
                veh_info = {}
                for key in veh:
                    veh_info[key.split("@")[-1]] = veh[key]
                veh_id = veh_info["id"]
                vehicles_dict[veh_id] = veh_info
            vehicle_trajs[str(round(float(time),1))] = vehicles_dict
    return vehicle_trajs

def get_line(fp, line_number):
    for i, x in enumerate(fp):
        if i == line_number:
            return x
    return None

def json2dict(jsonobj):
    vehicle_trajs = {}
    for time_step_info in jsonobj["fcd-export"]["timestep"]:
        time = time_step_info["@time"]
        vehicles_dict = {}
        for veh in time_step_info["vehicle"]:
            veh_info = {}
            for key in veh:
                veh_info[key.split("@")[-1]] = veh[key]
            veh_id = veh_info["id"]
            vehicles_dict[veh_id] = veh_info
        vehicle_trajs[str(round(float(time),1))] = vehicles_dict
    return vehicle_trajs

def update_vehicle_real_states(original_states, action, parameters, duration):
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
    phi = original_states[3] # vehicle heading in absolute coordinate
    phid = math.pi/2
    v = original_states[4] # lateral speed in vehicle coordinate
    r = original_states[5] # yaw rate in absolute coordinate
    whole_states = [original_states]
    Caf, Car = parameters["Caf"], parameters["Car"]
    a, b, m, Iz = parameters["a"], parameters["L"]-parameters["a"], parameters["m"], parameters["Iz"]
    # Use Runge–Kutta method
    k1 = helper_state_update(original_states, action, parameters)
    states_k2 = [original_states[i]+duration*k1[i]/2 for i in range(len(original_states))]
    k2 = helper_state_update(states_k2, action, parameters)
    states_k3 = [original_states[i]+duration*k2[i]/2 for i in range(len(original_states))]
    k3 = helper_state_update(states_k3, action, parameters)
    states_k4 = [original_states[i]+duration*k3[i] for i in range(len(original_states))]
    k4 = helper_state_update(states_k4, action, parameters)
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
    return RK_states
    

def helper_state_update(states, action, parameters):
    au = action["acceleration"] # longitudinal acceleration in vehicle coordinate 
    deltaf = action["steering_angle"]/180*math.pi # steering angle in vehicle coordinate
    u = states[0] # longitudinal speed in vehicle coordinate
    x = states[1] # longitudinal position in road coordinate
    y = states[2] # lateral position in road coordinate
    phi = states[3] # vehicle heading in absolute coordinate
    phid = math.pi/2
    v = states[4] # lateral speed in vehicle coordinate
    r = states[5] # yaw rate in absolute coordinate
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


def check_network_boundary(center_pos, restrictions, states, init_pos):
    within_flag = False
    x, y = center_pos
    x_lim, y_lim = restrictions
    new_states = list(states)
    new_center_pos = list(center_pos)
    # now only consider straight road in x direction
    if y > y_lim[1]:
        # print("find")
        new_center_pos[1] = y_lim[1]
        new_states[2] = y_lim[1] - init_pos[1]
        new_states[3] = math.pi/2
        new_states[4] = 0.
        new_states[5] = 0.
    elif y < y_lim[0]:
        # print("find")
        new_center_pos[1] = y_lim[0]
        new_states[2] = y_lim[0] - init_pos[1]
        new_states[3] = math.pi/2
        new_states[4] = 0.
        new_states[5] = 0.
    else:
        within_flag = True
    return new_center_pos, new_states, within_flag

def check_vehicle_info(veh_id, subsciption, simulator):
    # check lateral speed of BV in the beginning
    if veh_id != "CAV":
        heading = subsciption[67]
        v_lat = subsciption[50]
        if round(heading,2) != 90 and round(v_lat,2) == 0:
            # print(f"Zero lateral speed for heading!=90, so change lateral speed for {veh_id}!")
            subsciption[50] = (2*(heading > 90)-1)*(-4.)
        
    # check BV's angle especially during lane change for replay
    if veh_id != "CAV":
        heading = subsciption[67]
        if simulator.replay_flag and simulator.input is not None:
            time_step = str(round(float(simulator.env.init_time_step)+simulator.sumo_time_stamp,1))
            if time_step in simulator.input_trajs and veh_id in simulator.input_trajs[time_step]:
                if heading != float(simulator.input_trajs[time_step][veh_id]['angle']):
                    subsciption[67] = float(simulator.input_trajs[time_step][veh_id]['angle'])
                    # print(f"Not the same heading for {veh_id} at {time_step}, change from {heading} to {subsciption[67]}!")
    return subsciption


if __name__=="__main__":
    phid = math.pi/2
    new_state = update_vehicle_real_states(
        [35.08667107973639,67.59977027155756,0.2789663334258212,1.5179425592713829,0.9489990196176874,0.2428507294773491],
        {"acceleration":-2.0345304301208125,"steering_angle":5.876319125539895},
        {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500, # mass (kg)
            "Iz": 2420, # yaw moment of inertia (kg-m^2)
            "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
            "g": 9.81
        },
        0.1
    )
    print(new_state)
    print(-(new_state[4]*math.cos(new_state[3]-phid)+new_state[0]*math.sin(new_state[3]-phid)))
    new_state = update_vehicle_real_states(
        new_state,
        {"acceleration":-1.0893080234527588,"steering_angle":3.3162782192230225},
        {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500, # mass (kg)
            "Iz": 2420, # yaw moment of inertia (kg-m^2)
            "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
            "g": 9.81
        },
        0.1
    )
    print(new_state)
    print(-(new_state[4]*math.cos(new_state[3]-phid)+new_state[0]*math.sin(new_state[3]-phid)))
    new_state = update_vehicle_real_states(
        new_state,
        {"acceleration":-2.8639702796936035,"steering_angle":2.026840925216675,},
        {
            "L": 2.54, # wheel base (m)
            "a": 1.14, # distance c.g. to front axle (m)
            "m": 1500, # mass (kg)
            "Iz": 2420, # yaw moment of inertia (kg-m^2)
            "Caf": 44000*2, # cornering stiffness -- front axle (N/rad)
            "Car": 47000*2, # cornering stiffness -- rear axle (N/rad)
            "g": 9.81
        },
        0.1
    )
    print(new_state)
    print(-(new_state[4]*math.cos(new_state[3]-phid)+new_state[0]*math.sin(new_state[3]-phid)))
    