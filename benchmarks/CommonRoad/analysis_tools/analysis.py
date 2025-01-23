import numpy as np
import os
import matplotlib.pyplot as plt 
from scipy.stats import norm
import pandas as pd 
import json
import pickle
import argparse
# plt.style.use(["ggplot"])
# %config InlineBackend.figure_format = 'svg'

def analysis(root_folder):

    result_path = os.path.join(root_folder, 'statistical_results')
    result_data_path = os.path.join(result_path, "weight.npy")

    weighted_result_list = {'crash':[], 'infeasible':[], 'offroad':[], 'all':[]}

    files = sorted([f for f in os.listdir(root_folder) if f.startswith('weight_') and f.endswith('.json')])
    for file in files:
        file_path = os.path.join(root_folder, file)
        with open(file_path, 'r') as json_file:
            for line in json_file:
                episode_info = json.loads(line)
                weighted_result_list['all'].append(episode_info['crash_weight'])
                if episode_info['crash_weight'] > 0.9:
                    print(file_path, 'id:', episode_info['Episode'], 'weight:', episode_info['crash_weight'])
                if '1' in episode_info['reason']:
                    weighted_result_list['crash'].append(episode_info['crash_weight'])
                elif '10' in episode_info['reason']:
                    weighted_result_list['infeasible'].append(episode_info['crash_weight'])
                elif '6' in episode_info['reason']:
                    weighted_result_list['offroad'].append(episode_info['crash_weight'])

    os.makedirs(result_path, exist_ok=True)
    #np.save(result_data_path, np.array(weighted_result_list))
    print("Total episode number", len(weighted_result_list['all']))
    print("NADE crash number:", len(weighted_result_list['crash']))
    print("NADE infeasible number:", len(weighted_result_list['infeasible']))
    print("NADE crash rate:", np.sum(np.array(weighted_result_list['crash']))/len(weighted_result_list['all']))
    print("NADE infeasible rate:", np.sum(np.array(weighted_result_list['infeasible']))/len(weighted_result_list['all']))
    print("NADE offroad rate:", np.sum(np.array(weighted_result_list['offroad']))/len(weighted_result_list['all']))

    confidence_interval = 0.1
    z = norm.ppf(1-confidence_interval/2)

    original_result = pd.Series(weighted_result_list['all'])
    crash_mean_result = original_result.rolling(len(original_result), min_periods=1).mean()
    unit_std_result = original_result.rolling(len(original_result), min_periods=1).std()
    half_CI = z*unit_std_result/(np.sqrt(np.array(range(1, len(original_result)+1)))*crash_mean_result)
    half_CI_numpy = half_CI.to_numpy()
    crash_mean_result_numpy = crash_mean_result.to_numpy()
    #print("crash rate:", crash_mean_result_numpy[-1])
    print("RHW converge to 0.3 episode:", np.where(half_CI_numpy > 0.3)[0][-1])
    print("Mean = {:.10f}, Lower bound = {:.10f}, Higher bound = {:.10f}".format(crash_mean_result_numpy[-1], crash_mean_result_numpy[-1] * (1 - half_CI_numpy[-1]), crash_mean_result_numpy[-1] * (1 + half_CI_numpy[-1])))
    print("Final RHW:", half_CI_numpy[-1])

    with open(os.path.join(result_path,"result.txt"), "w") as fp:
        fp.write("Total episode number: {}\n".format(len(weighted_result_list['all'])))
        fp.write("NADE crash number: {}\n".format(len(weighted_result_list['crash'])))
        fp.write("NADE infeasible number: {}\n".format(len(weighted_result_list['infeasible'])))
        fp.write("NADE crash rate: {}\n".format(np.sum(np.array(weighted_result_list['crash']))/len(weighted_result_list['all'])))
        fp.write("NADE infeasible rate: {}\n".format(np.sum(np.array(weighted_result_list['infeasible']))/len(weighted_result_list['all'])))
        fp.write("NADE offroad rate: {}\n".format(np.sum(np.array(weighted_result_list['offroad']))/len(weighted_result_list['all'])))
        fp.write("RHW converge to 0.3 episode: {}\n".format(np.where(half_CI_numpy > 0.3)[0][-1]))
        fp.write("Mean = {:.10f}, Lower bound = {:.10f}, Higher bound = {:.10f}\n".format(crash_mean_result_numpy[-1], crash_mean_result_numpy[-1] * (1 - half_CI_numpy[-1]), crash_mean_result_numpy[-1] * (1 + half_CI_numpy[-1])))
        fp.write("Final RHW: {}\n".format(half_CI_numpy[-1]))

    fig = plt.figure(figsize=(8,6), dpi=100)
    plt.ylim(0,8e-3)
    plt.plot(crash_mean_result_numpy)
    plt.fill_between(range(len(crash_mean_result_numpy)), (1-half_CI_numpy)*crash_mean_result_numpy, (1+half_CI_numpy)*crash_mean_result_numpy, color=(229/256, 204/256, 249/256), alpha=0.9)
    plt.plot([0,len(crash_mean_result_numpy)],[5.92e-3]*2,"--",label="Base Model: 5.92e-03", alpha=0.5)
    plt.plot([0,len(crash_mean_result_numpy)],[4.77e-4]*2,"--",label="Base Model with Provable Safety Shield: 4.77e-04", alpha=0.5)
    plt.plot([0,len(crash_mean_result_numpy)],[crash_mean_result_numpy[-1]]*2,"--",label=f"Base Model with SafeDriver: {crash_mean_result_numpy[-1]:.2e}", alpha=0.5)

    plt.xlabel("Evaluation Number",fontsize=18)
    plt.ylabel("Crash Rate in NDE",fontsize=18)
    ax = fig.gca()
    ax.ticklabel_format(style='sci', scilimits=(-10,-8), axis='y')
    ax.ticklabel_format(style='sci', scilimits=(-10,-8), axis='x')
    tx = ax.xaxis.get_offset_text()
    tx.set_fontsize(12)
    tx = ax.yaxis.get_offset_text()
    tx.set_fontsize(12)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.legend()
    plt.savefig(os.path.join(result_path,"crash_rate.png"))