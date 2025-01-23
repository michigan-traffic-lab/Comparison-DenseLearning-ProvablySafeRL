import pickle, os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src.auxiliary.ReachableSetManager import ReachableSetManager
from src.auxiliary.safety import safety_shield, PlanningInfeasibleError
from src.sets.Polytope import Polytope
from src.sets.Zonotope import Zonotope
from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from src.load_commonRoad_scenario import load_commonRoad_scenario, occupancy_to_obstacle
from src.auxiliary.Obstacle import Obstacle
from gym_commonroad_sumo.draw_reachable_set import draw
import time

class SafeReinforcementPlanner:
    """class representing a safe reinforcement learning motion planner"""

    def __init__(self, planner, visualize=False, solver='linear', grouptol=None):
        """class constructor"""

        # store unsafe motion planner object
        self.planner = planner
        self.reachable_set_manager = self.planner.reachable_set_manager

        # initialize object properties
        self.safe_actions = []
        self.visualize = visualize
        self.solver = solver
        self.grouptol = grouptol
        self.action_corrected = False

        try:
            from spot_predictor.spot_predictor import SPOTPredictor
        except:
            print('fail to import SPOT')
            SPOTPredictor = None
        # create SPOT stuff
        self.spot_predictor = SPOTPredictor() if SPOTPredictor is not None else None

        self.prediction_steps = None

    def reset(self, scenario: Scenario, planning_problem_set: PlanningProblem, path=None, draw_flag=False):
        """reset the scenario, planning problem set, and the path to save the reachable set figures"""
        
        self.static_obstacles = load_commonRoad_scenario(scenario, only_load_static=True)
        self.scenario = scenario
        self.planning_problem_set = planning_problem_set
        self.path = path
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)
        self.save_info = {'observation':[], 'R':[], 'obstacles':[], 'scenario':[], 'info':[], 'u':[], 'intended_u':[], 'alpha':[], 'unsafe_alpha':[]}
        self.draw_flag = draw_flag

    def set_prediction_steps(self, prediction_steps: int):
        self.prediction_steps = prediction_steps

    def plan(self, observation, scenario):
        """plan a safe trajectory based on the current observations"""

        # obtain control command from the reinforcement learning agent
        start_time1 = time.time()
        unsafe_alpha = self.planner.plan(observation)
        self.reachable_set_manager.factor2control(unsafe_alpha)
        time_unsafe = time.time() - start_time1

        # apply safety shield to project to the closest safe control command
        u, opt_time = self.safety_shield(unsafe_alpha, observation, scenario) # if failed, return unsafe u for visualization
        if self.info == 'safe' or self.info == 'corrected':
            self.safe_actions = u
        else:
            self.safe_actions = u
        # elif len(self.safe_actions) > 0:
        #    self.info += ', \n use the last safe action'
        # else:
        #    self.safe_actions = self.generate_emergency_action(observation, scenario, unsafe_alpha)
        #    self.info += ', \n use emergency action'
        start_time2 = time.time()
        self.intended_actions = self.reachable_set_manager.factor2control(unsafe_alpha)
        end_time2 = time.time()
        time_safe = opt_time + end_time2 - start_time2
        # save info
        self.save_info['intended_u'].append(self.intended_actions[0])
        self.save_info['unsafe_alpha'].append(unsafe_alpha)

        # continue to drive in infeasible cases by using the last safe action or emergency action
        # except PlanningInfeasibleError:
        #     opt_time = None
        #     if len(self.safe_actions) == 0:
        #         raise PlanningInfeasibleError("No feasible actions available!")

        return self.safe_actions.pop(0), self.intended_actions.pop(0), time_unsafe, time_safe

    def generate_emergency_action(self, observation, scenario, alpha):
        """generate an emergency action based on the current observations"""

        # TODO: use ego lanelet id (include lane changing) to determine the closest BV
        alpha_new = alpha
        alpha_new[0][1] = 0
        x_pos = observation['xPosition']
        y_pos = observation['yPosition']
        dis_list1 = []
        dis_list2 = []
        for o in scenario.obstacles:
            if abs(o.initial_state.position[1] - y_pos) < 2:
                if o.initial_state.position[0] > x_pos:
                    dis_list1.append(o.initial_state.position[0] - x_pos)
                else:
                    dis_list2.append(x_pos - o.initial_state.position[0])
        if len(dis_list2) == 0:
            alpha_new[0][0] = -1
        elif len(dis_list1) == 0:
            alpha_new[0][0] = 1
        elif min(dis_list1) < min(dis_list2):
            alpha_new[0][0] = -1
        else:
            alpha_new[0][0] = 1
        
        u = self.reachable_set_manager.factor2control(alpha_new)
        return [u[0]]

    def observation2obstacle(self, observation, scenario):
        """convert the observations to a list of polytopic obstacles"""

        # v'xPosition': x[0], 'yPosition': x[1], 'velocity': x[2], 'orientation': x[3], 'time': time

        # get shift and angle
        phi = -observation['orientation']
        position = np.array([[-observation['xPosition']], [-observation['yPosition']]])

        self.prediction_dict = {}
        if self.spot_predictor is not None:
            # compute set-based predictions for dynamic obstacles
            self.prediction_dict, opt_time = self.spot_predictor.predict(scenario=scenario, planning_problem_set=self.planning_problem_set,
                prediction_steps=self.prediction_steps)
        else:
            opt_time = 0

        start_time = time.time()
        obstacles = []

        for dynamic_obstacle in list(self.prediction_dict):
            for o in dynamic_obstacle.prediction.occupancy_set:
                id = str(dynamic_obstacle.obstacle_id) + str(o.time_step)
                obstacles.append(occupancy_to_obstacle(o, self.scenario.dt, id=id))

        obstacles = obstacles + self.static_obstacles

        # transform obstacles to the local coordinate frame
        obs = []
        for o in obstacles:
            set = deepcopy(o.set)
            set.shift(position)
            set.rotate(phi)
            #time = o.time
            # deleted because all the obstacles' initial time step is set to 0 
            # if time is not None:
            #     time + (-observation['time'])
            obs.append(Obstacle(set, o.time))

        opt_time += time.time() - start_time

        return obs, opt_time
    
    def safety_shield(self, unsafe_alpha, observation, scenario):
        """project the action returned by the reinforcement learning agent to the closest safe action"""

        self.action_corrected = False
        start_time1 = time.time()
        # select the correct reachable set template based on the current velocity
        R = self.reachable_set_manager.select_reachable_set(observation)
        end_time1 = time.time()
        # get obstacles from the current observations
        obstacles, opt_time = self.observation2obstacle(observation, scenario)
        start_time2 = time.time()
        # apply the safety shield to obtain a control input that is guaranteed to be safe
        alpha_new, _, info = safety_shield(unsafe_alpha, R, obstacles, self.solver, self.grouptol)
        self.info = info
        if not np.allclose(unsafe_alpha, alpha_new):
            self.action_corrected = True

        # draw the current reachable set to check correctness
        if self.path is not None and self.draw_flag:
            try:
                time_step = int(observation['time']/scenario.dt)
                if time_step > 0:
                    draw(os.path.join(self.path, 'reachable_set', f'step_{time_step}.png'), observation, scenario, R, obstacles, info)
            except:
                pass

        # convert from factor vector alpha to control input u
        u = self.reachable_set_manager.factor2control(alpha_new)
        end_time2 = time.time()

        # save info
        self.save_info['observation'].append(observation)
        self.save_info['R'].append(R)
        self.save_info['obstacles'].append(obstacles)
        self.save_info['info'].append(info)
        self.save_info['scenario'].append(scenario)
        self.save_info['u'].append(u[0])
        self.save_info['alpha'].append(alpha_new)

        # visualize planned trajectory
        if self.visualize:
            self.visualization(alpha_new, R, obstacles, observation)

        return u, opt_time + end_time1 - start_time1 + end_time2 - start_time2

    def visualization(self, alpha, R, obstacles, observations):
        """visualize the planned trajectory"""

        plt.cla()

        # plot planned trajectory
        for o in deepcopy(R.occ):

            c = o.c + np.expand_dims(np.sum(o.G * np.prod(alpha**o.E, axis=0), axis=1), axis=1)
            Z = Zonotope(c, o.Grest)
            Z.plot('b')

        # plot environment
        try:
            self.planner.visualization(observations)
        except:
            test = 1

        # plot obstacles
        axes = plt.gca()
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()

        C = np.concatenate((np.identity(2), -np.identity(2)), axis=0)
        d = np.array([x_max + 1, y_max + 1, -x_min + 1, -y_min + 1])

        poly = Polytope(C, np.expand_dims(d, axis=1))

        for o in obstacles:
            try:
                (o.set.intersection(poly)).plot('r')
            except:
                test = 1

        plt.axis('equal')
        if 'time' in observations:
            plt.title(f"time={observations['time']}")
        plt.pause(0.1)
