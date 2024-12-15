'''
@Author: Clarence
@Date: 2024-10-8
@Description: This script is used to fill in the metamaterial for every stl file in the given urdf file folder.
'''

import os
import argparse
import subprocess
import sys
import time
import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("Project dir: ", project_dir)
sys.path.append(project_dir)

from metamaterial_filling.script.user_stl_metamaterial_filling_with_tenon import run_metamaterial_filling_for_stl_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fill in the metamaterial for every stl file in the given urdf file folder.')
    # parser.add_argument('--urdf_folder_name', type=str, default='gold_lynel20241010-134328_good', help='The folder of the urdf file.')
    # parser.add_argument('--opt_result_pkl_name', type=str, default='gold_lynel20241010-134332_robot_result.pkl', help='The pickle file name for the optimization result.')

    parser.add_argument('--urdf_folder', type=str, default=project_dir + '/result/gold_lynel_20241201-134522_good/result_round1/urdf', help='Input STL files folder path')
    parser.add_argument('--pkl_result_path', type=str, default=project_dir+'/result/gold_lynel_20241201-134522_good/result_round1/robot_result.pkl', help='Pickle file path for the tenon position results')

    args = parser.parse_args()

    # urdf_folder = project_dir + "/urdf/" + args.urdf_folder_name
    # opt_result_pkl_path = project_dir + "/auto_design/results/" + args.opt_result_pkl_name

    urdf_folder = args.urdf_folder
    opt_result_pkl_path = args.pkl_result_path

    print("Working on metamaterial for urdf files in the folder: ", urdf_folder)
    print("Using the optimization result: ", opt_result_pkl_path)

    # Find all the stl files in the urdf folder
    stl_files = [f for f in os.listdir(urdf_folder) if f.endswith('.stl')]
    print("Found ", len(stl_files), " stl files in the folder: ", urdf_folder)

    # Fill in the metamaterial for each stl file
    for stl_file in tqdm.tqdm(stl_files):
        if "BODY" in stl_file or "FOREARM" in stl_file:
            relative_density = 0.05
            plate_interval = 20
            shell_generation_voxel_resolution = 1.5
            shell_thickness = 1
        elif "ARM" in stl_file or "TAIL" in stl_file:
            relative_density = 0.1
            plate_interval = 20
            shell_generation_voxel_resolution = 1
            shell_thickness = 1
        else:
            relative_density = 0.15
            plate_interval = 10
            shell_generation_voxel_resolution = 1
            shell_thickness = 1.5

        input_stl_path = urdf_folder + "/" + stl_file
        unit = "m"
        biased_tenon_length = 2.5
        use_existing_shell = False
        pkl_result_path = opt_result_pkl_path
        tenon_file_folder = project_dir + "/metamaterial_filling/tenon"
        preview = False

        output_stl_name = stl_file.split(".")[0] + "_plate" + str(plate_interval) + '_' + str(shell_thickness) + 'mm_density_' + str(relative_density) + '30' + '.stl'

        # Check if the output file already exists
        output_stl_path = project_dir + "/metamaterial_filling/data/output/" + output_stl_name
        if os.path.exists(output_stl_path):
            print("Output file already exists: ", output_stl_name)
            continue

        run_metamaterial_filling_for_stl_file(input_stl_path, unit, relative_density, shell_thickness, shell_generation_voxel_resolution, plate_interval, biased_tenon_length, output_stl_name, use_existing_shell, pkl_result_path, tenon_file_folder, preview)

        time.sleep(5)
