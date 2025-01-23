import math
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from gym_commonroad_sumo.NDE_RL_NADE.envs.nde import *
from gym_commonroad_sumo.NDE_RL_NADE.controller.nadecontroller import NADEBackgroundController
from gym_commonroad_sumo.NDE_RL_NADE.controller.nadeglobalcontroller import NADEBVGlobalController
from gym_commonroad_sumo.NDE_RL_NADE.controller.rlcontroller import RLControllerNew
from gym_commonroad_sumo.NDE_RL_NADE.controller.crcontroller import CRController
from gym_commonroad_sumo.NDE_RL_NADE.nadeinfoextractor import NADEInfoExtractor


class NADE(NDE):
    def __init__(self,
        BVController=NADEBackgroundController,
        cav_model="RL",
        ndd_local_flag=True
        ):
        if cav_model == "RL":
            cav_controller = AVController
        elif cav_model == "RLNew":
            cav_controller = RLControllerNew
        elif cav_model == "IDM":
            cav_controller = IDMController
        elif cav_model == "Surrogate":
            cav_controller = SurrogateIDMAVController
        elif cav_model == "CR":
            cav_controller = CRController
        else:
            raise ValueError("Unknown AV controller!")
        super().__init__(
            AVController=cav_controller,
            BVController=BVController,
            AVGlobalController=DummyGlobalController,
            BVGlobalController=NADEBVGlobalController,
            info_extractor=NADEInfoExtractor,
            local_flag=ndd_local_flag
        )
        self.initial_weight = 1

    # @profile
    def _step(self, drl_action=None):
        """NADE subscribes all the departed vehicles and decides how to control the background vehicles.
        """
        # for vid in self.departed_vehicle_id_list:
        #     self.simulator.subscribe_vehicle(vid)
        self.drl_action = None
        super()._step(action=drl_action)
        self.drl_action = None


class NADE_ACM(NADE):
    def initialize(self):
        self.episode_info = {"id": self.simulator.episode, "start_time": self.simulator.get_time(), "end_time": None}
        # self.simulator.traci_step(20)
        self._cal_dist_init("route_0")
        self.generate_av(route="route_CAV",av_lane_id="-1006000.289.31_2",controller_type=ACMAVIDMController, position=0)
        if self.simulator.track_cav:
            self.simulator.track_vehicle_gui()
            self.simulator.set_zoom(500)
    
    def _step(self):
        super()._step()
    
    # @profile
    def _terminate_check(self):
        collision = tuple(set(self.simulator.detected_crash()))
        print("Collision result", collision)
        reason = None
        stop = False
        additional_info = {}
        if bool(collision) and "CAV" in collision:
            reason = {1: "CAV and BV collision"}
            stop = True
            additional_info = {"collision_id": collision}
        elif "CAV" not in self.vehicle_list:
            reason = {2: "CAV leaves network"}
            stop = True
        elif self.simulator.get_time() > 2000:
            reason = {3: "BVs leave network"}
            stop = True
        elif self.simulator.get_cav_travel_distance() > 7200:
            reason = {4: "CAV travels for 7200m"}
            stop = True
        if stop:
            self.episode_info["end_time"] = self.simulator.get_time()-self.simulator.step_size
        return reason, stop, additional_info

    def _cal_dist_init(self, routeID):
        re = {}
        route_edges = self.simulator.get_route_edges(routeID)
        init_edge = route_edges[0]
        for edge in route_edges:
            re[edge] = {}
            length = self.simulator.get_edge_length(edge)
            re[edge]["begin_distance"] = self.simulator.get_edge_dist(init_edge,0,edge,0)
            re[edge]["end_distance"] = self.simulator.get_edge_dist(init_edge,0,edge,length)
            re[edge]["available_lanes"] = self.simulator.get_available_lanes_id(edge)
        return re
