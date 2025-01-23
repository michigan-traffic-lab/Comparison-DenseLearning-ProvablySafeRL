import json
import os, glob, shutil

def split_replay_scenario(fcd_path, scenario_path, output_path, new_episode_baseline=20000):
    json_path = fcd_path.replace("fcd.json","json")
    fcd_info = []
    json_info = []
    ids = []
    with open(fcd_path) as fo:
        for line in fo:
            fcd_info.append(json.loads(line))
    with open(json_path) as fo:
        for line in fo:
            json_info.append(json.loads(line))
            ids.append(int(json_info[-1]['episode_info']['id'])+new_episode_baseline)
    for i in range(len(ids)):
        id_tmp = ids[i]
        fcd_tmp = fcd_info[i]
        json_tmp = json_info[i]
        os.makedirs(os.path.join(output_path, f"episode_{id_tmp}"), exist_ok=True)
        with open(os.path.join(output_path, f"episode_{id_tmp}", f"{id_tmp}.fcd.json"), "w") as fo:
            fo.write(json.dumps(fcd_tmp))
        with open(os.path.join(output_path, f"episode_{id_tmp}", f"{id_tmp}.json"), "w") as fo:
            fo.write(json.dumps(json_tmp))
        shutil.copy(os.path.join(scenario_path, f"episode_{id_tmp-new_episode_baseline}", "simulation_info.pkl"), os.path.join(output_path, f"episode_{id_tmp}", "simulation_info.pkl"))
        if os.path.exists(os.path.join(scenario_path, f"episode_{id_tmp-new_episode_baseline}", "planner_info.pkl")):
            shutil.copy(os.path.join(scenario_path, f"episode_{id_tmp-new_episode_baseline}", "planner_info.pkl"), os.path.join(output_path, f"episode_{id_tmp}", "planner_info.pkl"))
                
def transfer_all_replay_scenario(root_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    if root_path[-1] == "/":
        root_path = root_path[:-1]
    fcd_file_list = glob.glob(root_path+"/crash/*.fcd.json")
    for fcd_path in fcd_file_list:
        worker_id = int(fcd_path.split("/")[-1].replace(".fcd.json",""))
        scenario_path = root_path + f"/saved_data/worker_{worker_id}"
        split_replay_scenario(fcd_path, scenario_path, output_path, new_episode_baseline=worker_id*20000)