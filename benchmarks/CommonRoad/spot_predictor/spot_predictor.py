from commonroad.scenario.scenario import Scenario
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.scenario.trajectory import State

#from commonroad_spot.spot_interface import SPOTInterface
#from commonroad_spot.configurations import SPOTConfiguration
from spot_predictor.spot_interface import SPOTInterface
from omegaconf import OmegaConf


class SPOTPredictor:
    def __init__(self):

        config = OmegaConf.load("./spot_predictor/spot.yaml")
        spot_path = "./spot_predictor/spot"
        self.spot_interface = SPOTInterface(config=config, spot_path=spot_path)

        self.update_dict = {
            "Vehicle": {
                0: {  # 0 means that all obstacles will be changed
                    "a_max": 2,
                    "v_max": 45.0,
                    "compute_occ_m1": True, #spot_config.compute_assumption_m1,
                    "compute_occ_m2": True, #spot_config.compute_assumption_m2,
                    "compute_occ_m3": False, #spot_config.compute_assumption_m3,
                    "onlyInLane": config['only_in_lane'],
                }
            },
            "EgoVehicle": {
                0: {  # ID is ignored for ego vehicle
                    "a_max": 2,
                    "length": 5,
                    "width": 1.8,
                }
            }
        }

    # predict the occupancy of the scenario
    # Note: update_dict is hard to incorporate into SPOT. please adjust the parameters manually
    def predict(self, scenario: Scenario, planning_problem_set, prediction_steps: int):
        prediction_dict, opt_time = self.spot_interface.do_occupancy_prediction(
            scenario=scenario,
            planning_problem_set = planning_problem_set,
            prediction_horizon=prediction_steps*scenario.dt,
            update_dict=self.update_dict,
        )

        return prediction_dict, opt_time
