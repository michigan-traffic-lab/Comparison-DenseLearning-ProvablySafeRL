from gym_commonroad_sumo.simulation.video_render import create_video
import os, pickle
from gym_commonroad_sumo.file_reader import CommonRoadFileReader
import argparse

def visualize(root_folder):

    planning_problem_set = []

    # Load reset configuration for commonroad and sumo simulation
    def load_reset_config(path):
        fr = CommonRoadFileReader(os.path.join(path,'cr_map.xml'))
        scenario, planning_problem_set = fr.open()
        problem_dict = {'scenario': scenario, 'planning_problem_set': planning_problem_set}
        return problem_dict

    all_problem_dict = load_reset_config("./sumo_config/sumo_maps/3LaneHighway")
    planning_problem_set = all_problem_dict['planning_problem_set']

    data_folder = os.path.join(root_folder, 'saved_data')
    simulation_info_files = []
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file == 'simulation_info.pkl':
                simulation_info_files.append(os.path.join(root, file))

    for file in simulation_info_files:
        with open(file, 'rb') as f:
            simulation_info = pickle.load(f)
        output_folder_path = os.path.join(data_folder, 'videos')
        filename = file.split('/')[-3]+'_'+file.split('/')[-2]
        scenario = simulation_info['scenario']
        ego_vehicles = simulation_info['ego_vehicles']
        action_dict = simulation_info['action_dict']

        create_video(scenario,
                    output_folder_path,
                    planning_problem_set=planning_problem_set,
                    trajectory_pred=ego_vehicles,
                    follow_ego=True,
                    action_dict=action_dict,
                    filename=filename,
                    file_type='mp4')