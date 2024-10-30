import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
import trimesh
import tqdm

project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append(os.path.normpath(os.path.join(project_path, 'script')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design/modules')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'metamaterial_filling/script')))

from auto_design import auto_design_function
from quadruped_pose_to_pkl import Quadruped_Mesh_Transformer
from metamaterial_filling.script.pyansys_fea.mapdl_msh_analysis import MapdlFea

def quadruped_dataset_auto_design(args):
    # Find all stl files that contains "neutral" and "smoothed" in the dataset_path
    stl_files = []
    joint_npy_files = []
    for root, dirs, files in os.walk(args.dataset_path):
        for file in files:
            if "neutral" in file and "smoothed" in file and file.endswith(".stl"):
                joint_npy_file_path = file.replace("res_e300_smoothed.stl", "joints.npy")
                # Check if the joint file exists
                joint_npy_file_path = os.path.join(root, joint_npy_file_path)
                if os.path.exists(joint_npy_file_path):
                    joint_npy_files.append(joint_npy_file_path)
                    stl_files.append(os.path.join(root, file))

    print("Found {} stl files.".format(len(stl_files)))

    # Create result folder if not exists
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    # Do mesh transformation
    print("Start mesh transformation ...")
    joint_pkl_files = []
    transformed_stl_files = []
    for i in tqdm.tqdm(range(len(stl_files))):
        stl_file = stl_files[i]
        joint_npy_file = joint_npy_files[i]        

        mesh_transformer = Quadruped_Mesh_Transformer(stl_file, joint_npy_file, args.result_folder, 50)
        transformed_stl_file = mesh_transformer.get_result_stl_path()
        transformed_stl_files.append(transformed_stl_file)
        joint_plk_file = mesh_transformer.get_result_pkl_path()
        joint_pkl_files.append(joint_plk_file)

    # Do auto design
    print("Start auto design ...")
    result_folder = args.result_folder
    if args.do_fea_analysis:
        mapdl_object = MapdlFea()
    else:
        mapdl_object = None

    for i in tqdm.tqdm(range(len(joint_pkl_files))):
        print("Working on ", joint_pkl_files[i])
        args.joint_pkl_path = joint_pkl_files[i]
        args.stl_mesh_path = transformed_stl_files[i]
        args.result_folder = result_folder  # Update the result folder every time because it will be changed in the auto_design_function

        auto_design_function(args, mapdl_object)

    if args.do_fea_analysis:
        mapdl_object.shutdown()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    
    parser.add_argument('--dataset_path', type=str, default='/media/cc/DATA/anything2robot_data/standford_dogs/images/result', help='The path to the dataset.')
    parser.add_argument('--result_folder', type=str, default=os.path.normpath(project_path + '/result'), help='The folder to save the results.')

    parser.add_argument('--expected_x', type=float, default=22, help='The expected x-axis length of the model. (cm)')
    parser.add_argument('--voxel_size', type=float, default=1, help='The size of the voxel. (cm)')
    parser.add_argument('--voxel_density', type=float, default=1.2e-4, help='The estimated density of the voxel depending on the material. (kg/cm^3)')
    parser.add_argument('--joint_limitation', type=float, default=0.785, help='The limitation of the joint. +-joint_limitation. (rad)')

    parser.add_argument('--max_trial_round', type=int, default=8, help='The maximum number of trial rounds.')
    parser.add_argument('--genetic_generation', type=int, default=20, help='The number of generations for the genetic algorithm')
    parser.add_argument('--do_fea_analysis', type=bool, default=True, help='Do FEA analysis or not. If true, please make sure you have Ansys installed.')
    parser.add_argument('--regenerate_if_fea_failed', type=bool, default=True, help='Regenerate the model if the FEA analysis failed or not. FEAs are expensive and strict.')

    parser.add_argument('--visualize', type=bool, default=False, help='Visualize the process or not. Need to close the windows to continue the process if turned on.')
    parser.add_argument('--disable_joint_setting_ui', type=bool, default=True, help='Disable the joint setting UI or not. To continuously handle different models, this need to be set True.')
    parser.add_argument('--joint_setting_standard_scale', type=bool, default=False, help='Scale the model to a standard scale for easier joint setting in the UI or not')

    ### No need to set model_name. This is a temporary value. It will be removed in the future.
    parser.add_argument('--model_name', type=str, default='None', help='Temporary value. No need to set this value.')
    parser.add_argument('--stl_mesh_path', type=str, default='None', help='The path to the stl mesh file.')
    parser.add_argument('--joint_pkl_path', type=str, default='None', help='The path to the joint pkl file. Optional. If not provided, UI can be used to add joints.') 

    args = parser.parse_args()

    quadruped_dataset_auto_design(args)
    
