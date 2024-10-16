import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design/modules')))

#sys.path.append(os.path.normpath('../auto_design/modules'))

from modules.data_struct import *
from modules.mesh_loader import Custom_Mesh_Loader
from modules.mesh_decomp import Mesh_Decomp
from modules.motor_opt import Motor_Opt, Joint_Connect_Opt, get_bounds
from modules.interference_removal import InterferenceRemoval, RobotOptResult
from modules.destruction_check import destruction_check, destruction_check_urdf_folder

'''
Motor parameter library. This is used to store the motor parameters that is used in the optimization process.
'''
class MotorParameterLib:
    def __init__(self):
        self.tenon_height = 1.5 # cm. Tenon is the connection part between the motor and the link.

        # Height (cm), Radius (cm), Max Torque (N*M) 
        self.motor_lib = [[5.6 + self.tenon_height, 4.2, 12],   # DM6006. DAMIAO Tech         
                          [3.42 + self.tenon_height, 2.65, 2.5], # MG4005V2. K-Tech
                          [8.1 + self.tenon_height, 5.3, 20]]  # DM8009. DAMIAO Tech
        
        # This is the connector length between two motors in a 2 DOF joint. L shape. Unit: cm
        self.connector_lib = [[6, 6], 
                              [4, 4], 
                              [6, 6]]
    
    def get_motor_lib(self):
        return self.motor_lib
    
    def get_connector_lib(self):
        return self.connector_lib


'''
Auto design process
Check main function below to know how to use the function
'''
def auto_design(args):
    show_render_result = args.visualize

    # Load the mesh
    mesh_path = os.path.normpath(project_path + '/auto_design/model/given_models/' + args.model_name + '.stl')
    joint_path = os.path.normpath(project_path + '/auto_design/model/given_models/' + args.model_name + '_joints.pkl')
    
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_loader.load_mesh(mesh_path)
    mesh_loader.load_joint_positions(joint_path)
    expected_x = args.expected_x
    mesh_loader.scale(expected_x)

    avg_motor_cost_this = 1e6
    avg_motor_cost_threshold = 50  # A big number to filter out the insane results. No need to do mesh optimization if the motor cost is too high.
    enlarge_scale = 1.1
    
    counter = 0

    exit_code = -1

    # The motor cost should be less than the threshold
    while exit_code != 0 and counter < 5:
        counter += 1
        print("Decomposing and motor optimization process: ", counter)    

        # Do mesh decomposition
        print("Decomposing the mesh...")
        mesh_decomp = Mesh_Decomp(args, mesh_loader)
        mesh_decomp.decompose()
        if show_render_result:
            mesh_decomp.render()

        # Do actuator optimization. The first step in Motor_Opt will use pinocchio to calculate the torque and choose the motor.
        print("Optimizing the actuators...")
        bounds = get_bounds(mesh_decomp.link_tree, threshold=6)

        motor_param_lib = MotorParameterLib()
        motor_lib = motor_param_lib.get_motor_lib()
        connector_lib = motor_param_lib.get_connector_lib()
        
        motor_opt = Motor_Opt(args, mesh_decomp, bounds, motor_lib, connector_lib)
        motor_results, cost_log, best_fitness = motor_opt.run_opt(generation_num=args.genetic_generation)

        print("cost_log: ", cost_log)  #cost_log:  [(Motor_position_cost, Occupancy_cost), ...]
        print("Auto design best fitness: ", best_fitness)

        if show_render_result:
            motor_opt.render()
        time.sleep(1)

        # Up scale the mesh if the avg motor cost is too high
        avg_motor_cost_this = best_fitness / len(motor_results)
        print("Average motor cost: ", avg_motor_cost_this)
        if avg_motor_cost_this > avg_motor_cost_threshold:
            print("Failure Code 1. The motor cost is too high. Re-optimizing with a larger model... Scale the model by ", enlarge_scale)
            expected_x = expected_x * enlarge_scale
            mesh_loader.scale(expected_x)
            exit_code = 1
            continue
        else:
            exit_code = 0

        # Refine the mesh to connect the joints
        print("Refining the mesh to connect the actuators...")
        joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
        joint_connect_opt.run_opt()
        if show_render_result:
            mesh_decomp.mesh_group.render()
        time.sleep(3)

        # Remove the interference between the links while moving the joints
        print("Removing the interference between the links...")
        interference_removal = InterferenceRemoval(args=args, 
                                                mesh_group=mesh_decomp.mesh_group, 
                                                motor_param_result=motor_results, 
                                                link_tree=mesh_decomp.link_tree, 
                                                father_link_dict=mesh_decomp.father_link_dict)
        
        joint_limits = np.vstack([np.array([-args.joint_limitation, args.joint_limitation]) for _ in range(2*len(motor_results))])
        # joint_limits = np.vstack([np.array([-0.785, 0.785]) for _ in range(2*len(motor_results))])

        interference_removal.set_joint_limit(joint_limits)
        interference_removal.remove_interference()

        if show_render_result:
            interference_removal.mesh_group.render()

        time.sleep(1)
        urdf_path = interference_removal.generate_urdf()
        print("Saving the results... URDF file is saved at: ", urdf_path)

        # Save results
        robot_result = RobotOptResult(interference_removal, urdf_path, motor_lib)
        results_dir = './auto_design/results'
        os.makedirs(results_dir, exist_ok=True)
        pkl_file_path = os.path.normpath(results_dir + '/' + args.model_name + time.strftime("%Y%m%d-%H%M%S") + '_robot_result.pkl')
        pkl.dump(robot_result, open(pkl_file_path, 'wb'))

        # Run mesh destruction checking
        print("Checking the mesh destruction...")

        urdf_folder = os.path.dirname(urdf_path)
        destruction_check_pass = destruction_check_urdf_folder(urdf_folder, pkl_file_path, plotting=False)
        
        if not destruction_check_pass:
            print("Failure Code 2. The mesh is destroyed. Re-optimizing with a larger model... Scale the model by ", enlarge_scale)
            exit_code = 2
            continue
        else:
            exit_code = 0
        
        
        #TODO: Do FEA analysis for each link
        # if args.do_fea_analysis:


    return exit_code


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='gold_lynel', help='The model name. The model is expected to be placed in the model/given_models folder. (e.g. gold_lynel). The model should be in STL format.')
    parser.add_argument('--expected_x', type=float, default=50, help='The expected x-axis length of the model. (cm)')
    parser.add_argument('--voxel_size', type=float, default=1, help='The size of the voxel. (cm)')
    parser.add_argument('--voxel_density', type=float, default=1.5e-4, help='The estimated density of the voxel depending on the material. (kg/cm^3)')
    parser.add_argument('--joint_limitation', type=float, default=1, help='The limitation of the joint. +-joint_limitation. (rad)')

    parser.add_argument('--genetic_generation', type=int, default=5, help='The number of generations for the genetic algorithm')
    parser.add_argument('--do_fea_analysis', type=bool, default=False, help='Do FEA analysis or not. If true, please make sure you have Ansys installed.')
    parser.add_argument('--visualize', type=bool, default=True, help='Visualize the process or not')
    args = parser.parse_args()

    auto_design(args)
    