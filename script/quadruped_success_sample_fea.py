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

from metamaterial_filling.script.user_stl_force_relative_density_fea_opt import stl_force_relative_density_fea_opt

# success_flag, best_relative_density, recorded_relative_density, recorded_von_mises, recorded_displacement_magnitude = stl_force_relative_density_fea_opt(stl_path_input=stl_file, robot_result_file=pkl_file_path, check_only=False, max_iteration=max_iteration, display_fea_result=args.visualize, display_force_result=False, mapdl_object=mapdl_object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped Success Sample FEA")
    parser.add_argument("--ori_result_folder", type=str, default='/home/cc/git/anything2robot_result/result_2024Dec', help="Path to the result folder")
    parser.add_argument("--save_folder", type=str, default='/home/cc/git/anything2robot_result/fea_result_2024Dec', help="Path to save the fea result")
    args = parser.parse_args()

    # Start the fea object
    mapdl_object = MapdlFea()

    # Settings
    check_only=False
    max_iteration=5
    display_fea_result=False
    display_force_result=False

    # Create a csv file to save the overall fea result
    if not os.path.exists(os.path.join(args.save_folder, 'fea_result.csv')):
        with open(os.path.join(args.save_folder, 'fea_result.csv'), 'w') as f:
            f.write('robot_name,round_id,stl_file,success_flag,best_relative_density\n')

    # Iterate through all the child folders in the ori_result_folder
    valid_num = 0
    for child_folder in os.listdir(args.ori_result_folder):
        child_folder_path = os.path.join(args.ori_result_folder, child_folder)
        if os.path.isdir(child_folder_path):            
            # Iterate through all the child folders in the child_folder_path. Each child folder is named as 'result_roundi', find the one with the highest i
            max_i = 0
            for sub_folder in os.listdir(child_folder_path):
                if 'result_round' in sub_folder:
                    i = int(sub_folder.split('_')[-1][-1])
                    if i > max_i:
                        max_i = i
            
            max_i_folder_path = os.path.join(child_folder_path, f'result_round{max_i}')

            # Check if roundi_variable_exit_code_0.pkl exists
            pkl_file_path = os.path.join(max_i_folder_path, f'round{max_i}_variable_exit_code_0.pkl')
            if os.path.exists(pkl_file_path):
                # Run the fea
                valid_num += 1
                print(f"Running FEA for {max_i_folder_path}")

                # Create a folder to save the fea result
                fea_result_folder_path = os.path.join(args.save_folder, child_folder, f'result_round{max_i}')
                os.makedirs(fea_result_folder_path, exist_ok=True)

                # Iterate through all the stl files in the urdf folder and run the fea
                stl_folder_path = os.path.join(max_i_folder_path, 'urdf')
                for stl_file in os.listdir(stl_folder_path):
                    if stl_file.endswith('.stl'):
                        stl_file_path = os.path.join(stl_folder_path, stl_file)
                        print(f"Running FEA for {stl_file}")

                        result_file_path = os.path.join(fea_result_folder_path, f'{stl_file}_fea_result.pkl')
                        if os.path.exists(result_file_path):
                            print(f"FEA result already exists for {stl_file}")
                            continue    

                        # Run the fea
                        pkl_file_path = os.path.join(max_i_folder_path, 'robot_result.pkl')
                        success_flag, best_relative_density, recorded_relative_density, recorded_von_mises, recorded_displacement_magnitude = stl_force_relative_density_fea_opt(stl_path_input=stl_file_path, robot_result_file=pkl_file_path, check_only=check_only, max_iteration=max_iteration, display_fea_result=display_fea_result, display_force_result=display_force_result, mapdl_object=mapdl_object)
                        print(f"Success flag: {success_flag}, Best relative density: {best_relative_density}")
                        
                        # Save the fea result
                        with open(result_file_path, 'wb') as f:
                            fea_result_dict = {
                                'robot_name': child_folder,
                                'stl_file': stl_file,
                                'success_flag': success_flag,
                                'best_relative_density': best_relative_density,
                                'recorded_relative_density': recorded_relative_density,
                                'recorded_von_mises': recorded_von_mises,
                                'recorded_displacement_magnitude': recorded_displacement_magnitude
                            }
                            pkl.dump(fea_result_dict, f)

                        # Write the fea result to the csv file
                        with open(os.path.join(args.save_folder, 'fea_result.csv'), 'a') as f:
                            f.write(f"{child_folder},{max_i},{stl_file},{success_flag},{best_relative_density}\n")

    # Stop the fea object                   
    mapdl_object.shutdown()
                        
    print(f"Valid number: {valid_num}")
            
                        

