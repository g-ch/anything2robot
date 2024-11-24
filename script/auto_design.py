import os
import argparse
import numpy as np
import pickle as pkl
import time
# Add dependencies path
import sys
import trimesh
import time

project_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'auto_design/modules')))
sys.path.append(os.path.normpath(os.path.join(project_path, 'metamaterial_filling/script')))

#sys.path.append(os.path.normpath('../auto_design/modules'))

from modules.data_struct import *
from modules.mesh_loader import Custom_Mesh_Loader
from modules.mesh_decomp import Mesh_Decomp
from modules.motor_opt import Motor_Opt, Joint_Connect_Opt, get_bounds
from modules.interference_removal import InterferenceRemoval, RobotOptResult
from modules.destruction_check import destruction_check, destruction_check_urdf_folder

# Comment out the following two lines if you don't have Ansys installed
from metamaterial_filling.script.user_stl_force_relative_density_fea_opt import stl_force_relative_density_fea_opt
from metamaterial_filling.script.pyansys_fea.mapdl_msh_analysis import MapdlFea

'''
Motor parameter library. This is used to store the motor parameters that is used in the optimization process.
'''
class MotorParameterLib:
    def __init__(self):
        self.tenon_height = 1.5 # cm. Tenon is the connection part between the motor and the link.

        # Height (cm), Radius (cm), Max Torque (N*M) 
        self.motor_lib = [        
                          #[2.25, 2.15, 0.9], # GIM3505. SITAIWEI
                          [3.42, 2.65, 2.5], # MG4005V2. K-Tech
                          [3.65, 3.8, 12],   # DM6006. DAMIAO Tech 
                          [4.0, 4.8, 20], # DM8006. DAMIAO Tech
                          [6.2, 5.56, 120], # DM10010. DAMIAO Tech
                         ]  
        
        # TODO: CONSIDER MOTOR WEIGHT

        # This is the connector length between two motors in a 2 DOF joint. L shape. Unit: cm
        self.connector_lib = []
        for i in range(len(self.motor_lib)):
            # connector_length is radius + height/2 + 0.5. 0.5 is a margin.
            connector_length = self.motor_lib[i][1] + self.motor_lib[i][0]/2 + 0.5
            self.connector_lib.append([connector_length, connector_length])

        # Add self.tenon_height to the motor height to consider the tenon height
        for i in range(len(self.motor_lib)):
            self.motor_lib[i][0] += self.tenon_height
    
    def get_motor_lib(self):
        return self.motor_lib
    
    def get_connector_lib(self):
        return self.connector_lib


'''
This class is used to log the process of the optimization. The log contains a txt file and a pkl file to store the variables.
Use log() to log the string and add_variable() to add a variable to the log.
When the optimization is done, use close() to close the log file and save the variables.
'''
class Logger:
    def __init__(self, log_folder, model_name, round=1):
        self.model_name = model_name
        self.round = round
        self.exit_code = 0

        self.log_txt_path = log_folder + '/' + 'round' + str(round) + '.txt'

        self.log_file = open(self.log_txt_path, 'w')

        self.variable_dict = {}
        self.variable_dict['model_name'] = model_name
        self.variable_dict['round'] = round

    def log_variable(self, key, value, print_flag=True):
        self.variable_dict[key] = value
        if print_flag:
            print(key, ": ", value)
    
    def save_variable(self):
        self.variable_dict['exit_code'] = self.exit_code
        self.variable_log_file = self.log_txt_path.replace('.txt', '_variable_exit_code_' + str(self.exit_code) + '.pkl')
        pkl.dump(self.variable_dict, open(self.variable_log_file, 'wb'))
    
    def log_txt(self, log_str, print_flag=True):
        self.log_file.write(log_str + '\n')
        if print_flag:
            print(log_str)
    
    def close(self, exit_code=0):
        self.exit_code = exit_code
        self.save_variable()
        self.log_file.close()



def design_one_round(args, mesh_loader, round, log, round_result_saving_folder, enlarge_scale=1.1, avg_motor_cost_threshold=50, save_only=True, mapdl_object=None):
    exit_code = 0
    start_time = time.time()

    try:
        ##### Do mesh decomposition
        log.log_txt("Decomposing the mesh...")
        mesh_decomp_start_time = time.time()
        mesh_decomp = Mesh_Decomp(args, mesh_loader)
        mesh_decomp.decompose()
        mesh_decomp_end_time = time.time()

        # Log the number of voxels after the decomposition and show the result
        voxel_num = np.count_nonzero(mesh_decomp.mesh_group.voxel_data)
        log.log_variable('decompose_voxel_num', voxel_num)
        log.log_variable('decompose_time', mesh_decomp_end_time - mesh_decomp_start_time)
        decompose_result_image_path = round_result_saving_folder + '/decompose_result.png'
        mesh_decomp.render(save_only=save_only, save_path=decompose_result_image_path)

        ##### Do actuator optimization. The first step in Motor_Opt will use pinocchio to calculate the torque and choose the motor.
        log.log_txt("Optimizing the actuators...")
        actuator_opt_start_time = time.time()
        bounds = get_bounds(mesh_decomp.link_tree, threshold=6)
        motor_param_lib = MotorParameterLib()
        motor_lib = motor_param_lib.get_motor_lib()
        connector_lib = motor_param_lib.get_connector_lib()
        
        motor_opt = Motor_Opt(args, mesh_decomp, bounds, motor_lib, connector_lib)
        motor_results, cost_log, best_fitness = motor_opt.run_opt(generation_num=args.genetic_generation)
        actuator_opt_end_time = time.time()

        # Log the best fitness and the cost log of the optimization and show the result
        log.log_variable('motor_opt_cost_log', cost_log)
        log.log_variable('motor_opt_time', actuator_opt_end_time - actuator_opt_start_time)
        log.log_variable('motor_opt_motor_results', motor_opt.motor_results)
        log.log_txt("Auto design best fitness: " + str(best_fitness))
        motor_opt_image_path = round_result_saving_folder + '/motor_opt_result.png'
        motor_opt.render(save_only=save_only, save_path=None)

        # Up scale the mesh if the avg motor cost is too high
        avg_motor_cost_this = best_fitness / len(motor_results)
        log.log_txt("Average motor cost: " + str(avg_motor_cost_this))
        if avg_motor_cost_this > avg_motor_cost_threshold:
            log.log_txt("Failure Code 1. The motor cost is too high. Re-optimizing with a larger model... Scale the model by " + str(enlarge_scale))
            exit_code = 1
            log.close(exit_code)
            return exit_code

        ##### Refine the mesh to connect the joints
        log.log_txt("Refining the mesh to connect the actuators...")
        refine_start_time = time.time()
        joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
        joint_connect_opt.run_opt()
        refine_end_time = time.time()

        # Log the number of voxels after the joint connection and show the result
        voxel_num = np.count_nonzero(mesh_decomp.mesh_group.voxel_data)
        log.log_variable('joint_connect_voxel_num', voxel_num)
        log.log_variable('joint_connect_time', refine_end_time - refine_start_time)
        joint_connect_opt_image_path = round_result_saving_folder + '/joint_connect_opt_result.png'
        mesh_decomp.mesh_group.render(save_only=save_only, save_path=joint_connect_opt_image_path)

        ##### Remove the interference between the links while moving the joints
        log.log_txt("Removing the interference between the links...")
        interference_removal_start_time = time.time()
        interference_removal = InterferenceRemoval(args=args, 
                                                mesh_group=mesh_decomp.mesh_group, 
                                                motor_param_result=motor_results, 
                                                link_tree=mesh_decomp.link_tree, 
                                                father_link_dict=mesh_decomp.father_link_dict)
        # joint_limits = np.vstack([np.array([-0.785, 0.785]) for _ in range(2*len(motor_results))])
        interference_removal.set_joint_limit(args.joint_limitation, args.joint_limitation_from_champ)
        interference_removal.remove_interference()
        interference_removal_end_time = time.time()

        # Log the number of voxels after the interference removal and show the result
        voxel_num = np.count_nonzero(interference_removal.mesh_group.voxel_data)
        log.log_variable('interference_removal_voxel_num', voxel_num)
        log.log_variable('interference_removal_time', interference_removal_end_time - interference_removal_start_time)
        interference_removal_image_path = round_result_saving_folder + '/interference_removal_result.png'
        interference_removal.mesh_group.render(save_only=save_only, save_path=interference_removal_image_path)

        ##### Save results
        result_saving_start_time = time.time()
        urdf_path = interference_removal.generate_urdf(result_saving_folder=round_result_saving_folder)
        log.log_txt("Saving the results... URDF file is saved at: " + urdf_path)
        robot_result = RobotOptResult(interference_removal, urdf_path, motor_lib)
        
        pkl_file_path = os.path.normpath(round_result_saving_folder + '/robot_result.pkl')
        pkl.dump(robot_result, open(pkl_file_path, 'wb'))
        result_saving_end_time = time.time()

        log.log_variable('result_saving_time', result_saving_end_time - result_saving_start_time)

        ##### Run mesh destruction checking
        log.log_txt("Checking the mesh destruction...")
        destruction_check_start_time = time.time()
        urdf_folder = os.path.dirname(urdf_path)
        destruction_check_pass = destruction_check_urdf_folder(urdf_folder, pkl_file_path, plotting=False)
        destruction_check_end_time = time.time()
        
        log.log_variable('destruction_check_time', destruction_check_end_time - destruction_check_start_time)
        
        if not destruction_check_pass:
            log.log_txt("Failure Code 2. The mesh is destroyed. Re-optimizing with a larger model... Scale the model by " + str(enlarge_scale))
            exit_code = 2
            log.close(exit_code)
            return exit_code

        ##### Check if the meshes are watertight use trimesh
        water_tight_check_start_time = time.time()
        stl_files = []
        for root, dirs, files in os.walk(urdf_folder):  # Search the urdf folder to find all the stl files
            for file in files:
                if file.endswith(".stl"):
                    stl_files.append(os.path.join(root, file))
        
        for stl_file in stl_files:
            mesh = trimesh.load(stl_file)
            if not mesh.is_watertight:
                log.log_txt("Failure Code 3. The mesh is not watertight. Re-optimizing with a larger model... Scale the model by " + str(enlarge_scale))
                exit_code = 3
                break
        
        water_tight_check_end_time = time.time()
        log.log_variable('water_tight_check_time', water_tight_check_end_time - water_tight_check_start_time)
        
        if exit_code == 3:
            log.log_txt("The mesh is not watertight. Re-optimizing with a larger model... Scale the model by " + str(enlarge_scale))
            log.close(exit_code)
            return exit_code

        
        ##### Do FEA analysis for each link if the flag is set
        if args.do_fea_analysis:
            log.log_txt("Do FEA analysis...")
            fea_start_time = time.time()
            max_iteration = 2 # The maximum number of searching iterations for the FEA analysis to determine the best relative density of the voxels
            for stl_file in stl_files:
                if "BODY" in stl_file: # Skip the body part for quick testing
                    continue

                log.log_txt("*******Do FEA analysis for: " + stl_file)
                # More parameters can be set in the function stl_force_relative_density_fea_opt
                success_flag, best_relative_density = stl_force_relative_density_fea_opt(stl_path_input=stl_file, robot_result_file=pkl_file_path, check_only=True, max_iteration=max_iteration, display_fea_result=args.visualize, display_force_result=False, mapdl_object=mapdl_object)
                #exit()

                if not success_flag:
                    log.log_txt("Failure Code 4. The mesh is not feasible in FEA. Re-optimizing with a larger model... Scale the model by " + str(enlarge_scale))
                    exit_code = 4
                    break
                else:
                    file_name = os.path.basename(stl_file).split('.')[0]
                    log.log_txt("best_relative_density: " + str(best_relative_density))
                    log.log_variable('best_relative_density_' + file_name, best_relative_density)

                time.sleep(3)
            
            fea_end_time = time.time()
            log.log_variable('fea_time', fea_end_time - fea_start_time)
            
            if exit_code == 4 and args.regenerate_if_fea_failed:               
                log.close(exit_code)
                return exit_code
            else:
                log.log_txt("Success!!!!!!!!!!! The model is feasible in FEA.")
        
        
        log.log_txt("Round " + str(round) + " is done. Time: " + str(time.time() - start_time) + " seconds.")
        log.log_txt("Exit code: " + str(exit_code))
        log.close(exit_code)

    except Exception as e:
        log.log_txt("Error: " + str(e))
        log.log_txt("Error in round " + str(round) + ". Time: " + str(time.time() - start_time) + " seconds.")
        exit_code = 555
        log.close(exit_code)
        return exit_code
    
    return exit_code


'''
Check if the result already exists. If the result already exists, skip the optimization process.
'''
def check_if_result_exists(result_folder, model_name, max_round=8):
    result_exist = False
    subfolders = [f.path for f in os.scandir(result_folder) if f.is_dir()]

    for subfolder in subfolders:
        if model_name in subfolder:
            subsubfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
            if len(subsubfolders) >= max_round:
                result_exist = True
                return result_exist
            else:
                for subsubfolder in subsubfolders:
                    files = [f.path for f in os.scandir(subsubfolder) if f.is_file()]
                    for file in files:
                        if "exit_code_0" in file:
                            result_exist = True
                            return result_exist
    
    return result_exist



'''
Auto design process
@args: Check main function below to know what to pass in.
@mapdl_object: If the FEA analysis is turned on, an mapdl_object can be passed in to opening and turning off the Ansys Mapdl object outside the function. This is to avoid frequent opening and closing of the Ansys Mapdl object.
@return: The exit code. 0: Success. 1: The motor cost is too high. 2: The mesh is destroyed. 3: The mesh is not watertight. 4: The mesh is not feasible in FEA.
'''
def auto_design_function(args, mapdl_object=None):
    mesh_path = args.stl_mesh_path
    joint_path = args.joint_pkl_path

    model_name = os.path.basename(mesh_path).split('.')[0]
    args.model_name = model_name # To be used in other functions

    # Skip the optimization if the result already exists
    if check_if_result_exists(args.result_folder, model_name, args.max_trial_round):
        print("The result already exists. Skip the optimization process.")
        return 0

    args.result_folder = args.result_folder + '/' + model_name + '_' + time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.result_folder, exist_ok=True)

    # Check if the stl file exists. If yes, scale it to 50cm in x-axis for joint setting. We use 50 as a standard scale for joint setting.
    if not os.path.exists(mesh_path):
        print("Error: The mesh path doesn't exist.")
        return -1
    
    # Use a standard scale (delt x = 50cm) for joint setting if the flag is set
    if args.joint_setting_standard_scale:
        original_mesh = trimesh.load(mesh_path)
        bounds = original_mesh.bounds
        scale_factor = 50 / (bounds[1][0] - bounds[0][0])
        original_mesh.apply_scale(scale_factor)
        scaled_mesh_save_path = args.result_folder + '/scaled_model_for_joint_setting.stl'
        original_mesh.export(scaled_mesh_save_path)
        mesh_path = scaled_mesh_save_path
        args.stl_mesh_path = mesh_path

    # Check if the joint path exists. If not, the UI shouldn't be disabled.
    if not os.path.exists(joint_path):
        print("Warning: The joint path doesn't exist. The joint setting UI will be enabled.")
        args.disable_joint_setting_ui = False

    ##### Load the mesh and joint positions
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_loader.load_mesh(mesh_path)
    mesh_loader.load_joint_positions(joint_path, figure_save_path=args.result_folder + '/joint_positions.png')
    expected_x = args.expected_x
    
    # Set the motor cost threshold
    avg_motor_cost_threshold = 50  # A big number to filter out the insane results. No need to do mesh optimization if the motor cost is too high.
    enlarge_scale = 1.1 # This could also be a parameter to be set by the user.

    external_mapdl_object = False
    if args.do_fea_analysis:
        if mapdl_object is None:
            mapdl_object = MapdlFea() # Start the Ansys Mapdl object if the FEA analysis is turned on
        else:
            external_mapdl_object = True
    
    round = 0
    exit_code = -1

    save_only = True
    if args.visualize:
        save_only = False
    
    # The motor cost should be less than the threshold
    while exit_code != 0 and round < args.max_trial_round:
        round += 1

        # Make a subfolder to save the results of each round
        round_result_saving_folder = args.result_folder + '/result_round' + str(round)
        os.makedirs(round_result_saving_folder, exist_ok=True)

        # Create a log file to log the process
        log = Logger(round_result_saving_folder, model_name, round)

        # Scale the model. If it is not the first round, the model will be scaled further by enlarge_scale.
        if round > 1:
            expected_x = expected_x * enlarge_scale
        
        mesh_loader.scale(expected_x, save_path=round_result_saving_folder + '/scaled_model_expected_x_' + str(expected_x) + '.stl')

        log.log_txt("Decomposing and motor optimization process: " + str(round))    

        exit_code = design_one_round(args, mesh_loader, round, log, round_result_saving_folder, enlarge_scale=enlarge_scale, avg_motor_cost_threshold=avg_motor_cost_threshold, save_only=save_only, mapdl_object=mapdl_object)
        


    if args.do_fea_analysis and not external_mapdl_object: 
        # If the mapdl_object is created in this function, then we need to shut it down. Otherwise, we need to keep it alive.
        mapdl_object.shutdown()

    print("Finished with exit code: ", exit_code, " in ", round, " rounds.")
    print("The results are saved at: ", args.result_folder)

    return exit_code


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')

    parser.add_argument('--stl_mesh_path', type=str, default=os.path.normpath(project_path + '/auto_design/model/given_models/gold_lynel.stl'), help='The path to the stl mesh file.')
    parser.add_argument('--joint_pkl_path', type=str, default=os.path.normpath(project_path + '/auto_design/model/given_models/gold_lynel_joints.pkl'), help='The path to the joint pkl file. Optional. If not provided, UI can be used to add joints.') 
    
    # parser.add_argument('--stl_mesh_path', type=str, default="C:\\Users\\smogg\\OneDrive\\Desktop\\workspace\\anything2robot\\auto_design\\model\\given_models\\gold_lynel.stl")
    # parser.add_argument('--joint_pkl_path', type=str, default="C:\\Users\\smogg\\OneDrive\\Desktop\\workspace\\anything2robot\\auto_design\\model\\given_models\\gold_lynel_joints.pkl")

    parser.add_argument('--result_folder', type=str, default=os.path.normpath(project_path + '/result'), help='The folder to save the results.')

    parser.add_argument('--expected_x', type=float, default=50, help='The expected x-axis length of the model. (cm)')
    parser.add_argument('--voxel_size', type=float, default=1, help='The size of the voxel. (cm)')
    parser.add_argument('--voxel_density', type=float, default=1.2e-4, help='The estimated density of the voxel depending on the material. (kg/cm^3)')
    parser.add_argument('--joint_limitation', type=float, default=0.785, help='The limitation of the joint. +-joint_limitation. (rad)')
    parser.add_argument('--joint_limitation_from_champ', type=bool, default=True, help='Use champ controller or not. This will affect joint limits.')

    parser.add_argument('--max_trial_round', type=int, default=8, help='The maximum number of trial rounds.')
    parser.add_argument('--genetic_generation', type=int, default=10, help='The number of generations for the genetic algorithm')
    parser.add_argument('--do_fea_analysis', type=bool, default=False, help='Do FEA analysis or not. If true, please make sure you have Ansys installed.')
    parser.add_argument('--regenerate_if_fea_failed', type=bool, default=False, help='Regenerate the model if the FEA analysis failed or not. FEAs are expensive and strict.')

    parser.add_argument('--visualize', type=bool, default=True, help='Visualize the process or not. Need to close the windows to continue the process if turned on.')
    parser.add_argument('--disable_joint_setting_ui', type=bool, default=False, help='Disable the joint setting UI or not')
    parser.add_argument('--joint_setting_standard_scale', type=bool, default=False, help='Scale the model to a standard scale for easier joint setting in the UI or not')

    ### No need to set model_name. This is a temporary value. It will be removed in the future.
    parser.add_argument('--model_name', type=str, default='None', help='Temporary value. No need to set this value.')

    args = parser.parse_args()

    auto_design_function(args)
    