import matlab.engine
from gym_commonroad_sumo.file_writer import CommonRoadFileWriter
from gym_commonroad_sumo.file_reader import CommonRoadFileReader
from commonroad.common.file_writer import OverwriteExistingFile
import os, time

class SPOTInterface:
    def __init__(self, config, spot_path = None):
        self.config = config
        
        # start MATLAB engine
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(spot_path))

    # save current scenario to xml file and call SPOT to predict the occupancy
    def do_occupancy_prediction(self, scenario, 
                                planning_problem_set,
                                prediction_horizon,
                                update_dict,):
        path = "./tmp"
        os.makedirs(path, exist_ok=True)
        occupancy_prediction = []
        try:
            filename = self.write_scenario_to_xml(scenario, planning_problem_set, path)
            opt_time = self.eng.spot_main(filename, scenario.dt, prediction_horizon)
            occupancy_prediction = self.read_occupancy_xml(filename)
            os.remove(filename)
        except:
            opt_time = 0
            pass
        return occupancy_prediction, opt_time
    
    # write scenario to xml file
    def write_scenario_to_xml(self, scenario, planning_problem_set, path):
        fw = CommonRoadFileWriter(scenario, planning_problem_set, '', '', '', '')
        tmp_name = time.time()
        tmp_name = str(tmp_name).split('.')
        filename = os.path.join(path,"{}.xml".format(tmp_name[0]+tmp_name[1]))
        fw.write_to_file(filename, OverwriteExistingFile.ALWAYS)
        return filename
    
    # read occupancy from xml file
    def read_occupancy_xml(self, filename):
        fr = CommonRoadFileReader(filename)
        scenario, _ = fr.open()
        return scenario.obstacles


if __name__=="__main__":
    print("aaaaa")