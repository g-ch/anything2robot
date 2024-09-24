import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
sys.path.append(os.path.normpath('./auto_design/modules'))

from modules.data_struct import *
from modules.mesh_loader import Custom_Mesh_Loader
from modules.mesh_decomp import Mesh_Decomp
from modules.motor_opt import Motor_Opt, Joint_Connect_Opt, get_bounds
from modules.interference_removal import InterferenceRemoval, RobotOptResult

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=50, help='The expected width of the model')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='The size of the voxel')
    parser.add_argument('--voxel_density', type=float, default=2e-5, help='The density of the voxel')
    args = parser.parse_args()
    mesh_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '.stl')
    joint_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '_joints.pkl')
    
    
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_loader.load_mesh(mesh_dir)
    mesh_loader.load_joint_positions(joint_dir)
    mesh_loader.scale()
    mesh_decomp = Mesh_Decomp(args, mesh_loader)
    mesh_decomp.decompose()
    mesh_decomp.render()


    bounds = get_bounds(mesh_decomp.link_tree, threshold=6)
    motor_lib = [[3.6, 3.8, 12],   # DM6006         # Height, Radius, Torque
                #  [4.5, 2.5, 8 ], # DM4310
                 [6.1, 4.9, 20 ]]  # DM8009
    motor_opt = Motor_Opt(args, mesh_decomp, bounds, motor_lib)
    motor_results = motor_opt.run_opt()
    motor_opt.render()
    joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
    joint_connect_opt.run_opt()
    mesh_decomp.mesh_group.render()
    interference_removal = InterferenceRemoval(args=args, 
                                               mesh_group=mesh_decomp.mesh_group, 
                                               motor_param_result=motor_results, 
                                               link_tree=mesh_decomp.link_tree, 
                                               father_link_dict=mesh_decomp.father_link_dict)
    joint_limits = np.vstack([np.array([-0.3, 0.3]) for _ in range(2*len(motor_results))])
    interference_removal.set_joint_limit(joint_limits)
    interference_removal.remove_interference()
    interference_removal.mesh_group.render()
    urdf_dir = interference_removal.generate_urdf()

    # Save results
    robot_result = RobotOptResult(interference_removal, urdf_dir)
    results_dir = './auto_design/results'
    os.makedirs(results_dir, exist_ok=True)
    pkl.dump(robot_result, open('./auto_design/results/' + args.model_name + time.strftime("%Y%m%d-%H%M%S") + '_robot_result.pkl', 'wb'))