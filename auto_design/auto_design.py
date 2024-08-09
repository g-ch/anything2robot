import os
import argparse
import numpy as np

# Add dependencies path
import sys
sys.path.append(os.path.normpath('./auto_design/modules'))

from modules.data_struct import *
from modules.mesh_loader import Custom_Mesh_Loader
from modules.mesh_decomp import Mesh_Decomp
from modules.motor_opt import Motor_Opt, Joint_Connect_Opt, get_bounds

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=40, help='The expected width of the model')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='The size of the voxel')
    parser.add_argument('--voxel_density', type=float, default=1e-4, help='The density of the voxel')
    args = parser.parse_args()
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '.stl')
    joint_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '_joints.pkl')
    mesh_loader.load_mesh(mesh_dir)
    mesh_loader.load_joint_positions(joint_dir)
    mesh_loader.scale()

    mesh_decomp = Mesh_Decomp(args, mesh_loader)
    mesh_decomp.decompose()
    mesh_decomp.render()
    bounds = np.array(get_bounds(mesh_decomp.link_tree, threshold=6)).reshape(-1, 2)
    motor_lib = [[3.6, 3.8, 12],  # DM6006         # Height, Radius, Torque
                #  [4.5, 2.5, 8 ],  # DM4310
                 [3.75, 4.8, 20 ]]  # DM4310
    

    
    motor_opt = Motor_Opt(args, mesh_decomp, bounds, motor_lib)
    motor_results = motor_opt.run_opt()
    # np.save('./results/' + args.model_name + '_motor_results1.npy', motor_results)
    # motor_opt.render()
    
    # motor_opt = Motor_Opt(args, mesh_decomp, None, None)
    # motor_opt.motor_results = np.load('./auto_design/results/' + args.model_name + '_motor_results1.npy')
    # motor_opt.render()
    
    joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
    joint_connect_opt.run_opt()
    mesh_decomp.render()