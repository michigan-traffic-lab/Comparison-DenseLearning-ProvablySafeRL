from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.simulator import Simulator
from gym_commonroad_sumo.NDE_RL_NADE.mtlsp.controller.vehicle_controller.controller import Controller, BaseController


class CRController(BaseController):
    def __init__(
        self,
        subscription_method=Simulator.subscribe_vehicle_all_information,
        controllertype="CRController",
    ):
        super().__init__(
            subscription_method=subscription_method, controllertype=controllertype
        )
        self.state = None
        self.criticality = 0.0

    def install(self):
        if self.subscription_method:
            self.subscription_method(self.vehicle.id)
        if self.observation_method:
            self.vehicle.observation_method = self.observation_method