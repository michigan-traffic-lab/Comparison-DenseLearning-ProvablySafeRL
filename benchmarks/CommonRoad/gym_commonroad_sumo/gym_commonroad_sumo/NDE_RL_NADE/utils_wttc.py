import glob
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt

from controller.neuralmetric import NN_Metric

def load_json_file(path):
    with open(path) as fp:
        info = json.load(fp)
    return info

veh_length = 5
veh_width = 1.8

yaml_config_path = "./configs_NNMetric/randomsample_3model_0s_before_unavoidable_normalized_aw3.yaml"
new_neural_metric = NN_Metric(yaml_config_path)

def _cal_box_vertex(x,y,heading_vector,length=veh_length,width=veh_width):
    headleft = [
        x-width/2*heading_vector[1],
        y+width/2*heading_vector[0]
    ]
    headright = [
        x+width/2*heading_vector[1],
        y-width/2*heading_vector[0]
    ]
    tailleft = [
        x-length*heading_vector[0]-width/2*heading_vector[1],
        y-length*heading_vector[1]+width/2*heading_vector[0]
    ]
    tailright = [
        x-length*heading_vector[0]+width/2*heading_vector[1],
        y-length*heading_vector[1]-width/2*heading_vector[0]
    ]
    return [headleft, headright, tailleft, tailright]

def cal_wttc(traj_json):
    wttc_list = []
    traj_info = traj_json["fcd-export"]["timestep"]
    for step in traj_info:
        # if float(step["@time"]) >= 18.5:
        #     print()
        cav = step["vehicle"][-1]
        assert(cav["@id"]=="CAV")
        sv_headx, sv_heady = float(cav["@x"]), float(cav["@y"])
        sv_heading = float(cav["@angle"])/180*np.pi
        sv_heading = [np.sin(sv_heading), np.cos(sv_heading)] # sin and cos
        sv_centerx = sv_headx-veh_length/2*sv_heading[0]
        sv_centery = sv_heady-veh_length/2*sv_heading[1]
        sv_box = _cal_box_vertex(sv_centerx,sv_centery,sv_heading)
        sv_v = float(cav["@speed"])
        sv_pov_dist_list = []
        for veh in step["vehicle"]:
            if veh["@id"] != "CAV":
                pov_headx, pov_heady = float(veh["@x"]), float(veh["@y"]) 
                pov_heading = float(veh["@angle"])/180*np.pi
                pov_heading = [np.sin(pov_heading), np.cos(pov_heading)] # sin and cos
                pov_centerx = pov_headx-veh_length/2*pov_heading[0]
                pov_centery = pov_heady-veh_length/2*pov_heading[1]
                sv_pov_dist_list.append((sv_centerx-pov_centerx)**2+(sv_centery-pov_centery)**2)
        pov = step["vehicle"][sv_pov_dist_list.index(min(sv_pov_dist_list))]
        pov_headx, pov_heady = float(pov["@x"]), float(pov["@y"]) 
        pov_heading = float(pov["@angle"])/180*np.pi
        pov_heading = [np.sin(pov_heading), np.cos(pov_heading)] # sin and cos
        pov_centerx = pov_headx-veh_length/2*pov_heading[0]
        pov_centery = pov_heady-veh_length/2*pov_heading[1]
        pov_box = _cal_box_vertex(pov_centerx,pov_centery,pov_heading)
        pov_v = float(veh["@speed"])

        pairs = itertools.product(sv_box, pov_box)

        sv_x, sv_y = sv_centerx, sv_centery
        pov_x, pov_y = pov_centerx, pov_centery
        min_dist = (sv_centerx-pov_centerx)**2+(sv_centery-pov_centery)**2
        for p in pairs:
            _dist = (p[0][0]-p[1][0])**2+(p[0][1]-p[1][1])**2
            if _dist < min_dist:
                min_dist = _dist
                sv_x, sv_y = p[0]
                pov_x, pov_y = p[1]

        wttc = _cal_WTTC_helper(sv_x, sv_y, sv_heading, sv_v, pov_x, pov_y, pov_heading, pov_v)
        wttc_list.append(wttc)
    return wttc_list

def _cal_WTTC_helper(sv_x, sv_y, sv_heading, sv_v, pov_x, pov_y, pov_heading, pov_v, a1=1,a2=2):
    func_params = np.zeros(5)

    xa, ya = sv_x, sv_y
    vax, vay = sv_v * sv_heading[0], sv_v * sv_heading[1]

    xb, yb = pov_x, pov_y
    vbx, vby = pov_v * pov_heading[0], pov_v * pov_heading[1]

    func_params[0] = -1/4
    func_params[1] = 0
    func_params[2] = (vbx-vax)**2 / (a1**2) + (vby-vay)**2 / (a2**2)
    func_params[3] = 2 * (vbx-vax) * (xb-xa) / a1**2 + 2 * (vby-vay) * (yb-ya) / a2**2
    func_params[4] = (xb-xa)**2/a1**2 + (yb-ya)**2/a2**2

    sol = np.roots(func_params)
    wttc = np.real(np.min(sol[(np.imag(sol)==0)&(sol>0)])).item()
    return wttc

def plot_crit_wttc_newcrit(crit_list, wttc_list, new_crit_list, episode_id):
    plt.figure(dpi=400)
    font = {'family' : 'Times New Roman',
        'size'   : 15}
    plt.rc('font', **font)
    t1 = [t*0.1 for t in range(len(crit_list))]
    t2 = [t*0.1 for t in range(len(wttc_list))]
    t3 = [t*0.1 for t in range(len(new_crit_list))]
    plt.plot(t1, crit_list, label="neural metric\n(within 0.2s before unavoidable crash)")
    # plt.plot(t2, wttc_list, label="WTTC")
    plt.plot(t3, new_crit_list, label="neural metric\n(unavoidable crash)")
    plt.xlabel("time (s)")
    plt.ylabel("criticality")
    plt.legend()
    plt.title(episode_id)
    
    plt.savefig(f'test_{episode_id}.png', bbox_inches = 'tight', dpi=400)


if __name__=="__main__":
    folder = "/media/mtl/WD/DATA_HAOJIE/AVTraining/IterativeOfflineTraining_NewExp/OfflineCollecting/tmp/neuralmetric_iter1_new/test_videos_dm545bm369_neuralmetric02s/videos1/crash"
    for file in sorted(glob.glob(folder+"/*.fcd.json")):
        fcd_info = load_json_file(file)
        wttc_list = cal_wttc(fcd_info)
        json_info = load_json_file(file.replace("fcd.json","json"))
        crit_list = [json_info["CAV_info"][step]["criticality"] for step in json_info["CAV_info"]]
        new_crit_list = [new_neural_metric.inference(np.array(json_info["CAV_info"][step]["CAV_action"]["additional_info"]["NN_metric_obs"]).reshape(1,-1)) for step in json_info["CAV_info"]]
        plot_crit_wttc_newcrit(crit_list, wttc_list, new_crit_list, fcd_info["original_name"])

