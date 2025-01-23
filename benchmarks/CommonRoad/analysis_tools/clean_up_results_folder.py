import os
import json
import shutil
import argparse

def clean(root_folder):

    # Delete all files with '_' in their name in root_folder/crash
    crash_folder = os.path.join(root_folder, "crash")
    for filename in os.listdir(crash_folder):
        if '_' in filename:
            os.remove(os.path.join(crash_folder, filename))

    # Delete all empty folders in root_folder/saved_data
    saved_data_folder = os.path.join(root_folder, "saved_data")
    for _ in range(2):
        for dirpath, dirnames, filenames in os.walk(saved_data_folder, topdown=False):
            if not dirnames and not filenames:
                os.rmdir(dirpath)

    # Merge all weight_*.json files and modify the 'Episode' key
    with open(os.path.join(root_folder, "weight_all.json"), 'a') as new_file:
        id_list = []
        for filename in os.listdir(root_folder):
            if filename.startswith("weight_") and filename.endswith(".json") and filename != "weight_all.json":
                id = int(filename.split('.')[0].split('_')[1])
                id_list.append(id)
        id_list.sort()
        for id in id_list:
            filename = "weight_" + str(id) + ".json"
            with open(os.path.join(root_folder, filename), 'r') as f:
                for line in f:
                    data = json.loads(line)
                    #data['Episode'] = 'worker' + filename.split('.')[0].split('_')[1] + '_' + str(data['Episode'])
                    json.dump(data, new_file)
                    new_file.write('\n')
                os.remove(os.path.join(root_folder, filename))