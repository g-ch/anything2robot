import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
import trimesh

project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append(os.path.normpath(os.path.join(project_path, 'script')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design/modules')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'metamaterial_filling/script')))

from auto_design import auto_design_function


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')

    parser.add_argument('--stl_mesh_path', type=str, default=os.path.normpath(project_path + '/auto_design/model/given_models/gold_lynel.stl'), help='The path to the stl mesh file.')
    parser.add_argument('--joint_pkl_path', type=str, default=os.path.normpath(project_path + '/auto_design/model/given_models/gold_lynel_joints.pkl'), help='The path to the joint pkl file. Optional. If not provided, UI can be used to add joints.') 
    
    parser.add_argument('--result_folder', type=str, default=os.path.normpath(project_path + '/result'), help='The folder to save the results.')

    parser.add_argument('--expected_x', type=float, default=50, help='The expected x-axis length of the model. (cm)')
    parser.add_argument('--voxel_size', type=float, default=1, help='The size of the voxel. (cm)')
    parser.add_argument('--voxel_density', type=float, default=1.5e-4, help='The estimated density of the voxel depending on the material. (kg/cm^3)')
    parser.add_argument('--joint_limitation', type=float, default=1, help='The limitation of the joint. +-joint_limitation. (rad)')

    parser.add_argument('--max_trial_round', type=int, default=5, help='The maximum number of trial rounds.')
    parser.add_argument('--genetic_generation', type=int, default=5, help='The number of generations for the genetic algorithm')
    parser.add_argument('--do_fea_analysis', type=bool, default=False, help='Do FEA analysis or not. If true, please make sure you have Ansys installed.')
    parser.add_argument('--regenerate_if_fea_failed', type=bool, default=False, help='Regenerate the model if the FEA analysis failed or not. FEAs are expensive and strict.')

    parser.add_argument('--visualize', type=bool, default=False, help='Visualize the process or not. Need to close the windows to continue the process if turned on.')
    parser.add_argument('--disable_joint_setting_ui', type=bool, default=True, help='Disable the joint setting UI or not')

    ### No need to set model_name. This is a temporary value. It will be removed in the future.
    parser.add_argument('--model_name', type=str, default='None', help='Temporary value. No need to set this value.')

    args = parser.parse_args()

    auto_design_function(args)
    
