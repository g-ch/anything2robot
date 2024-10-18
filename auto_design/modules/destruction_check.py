'''
This script is used to check the destruction of mesh by using the tenon position of each mesh.
'''

import os
import numpy as np
import pickle as pkl
import time
import sys
import pyvista as pv
import argparse

this_file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(os.path.join(this_file_path, '..')))

from interference_removal import RobotOptResult

'''
Check if the mesh is destroyed in the interference removal
@param stl_path: The path to the STL file. Unit: m
@param result_pkl_path: The path to the result pkl file
@param voxel_size: The voxel size used in the voxelization. Unit: m
'''
def destruction_check(stl_path, result_pkl_path, voxel_size=0.005, plotting=True):  
    # Check if the input STL file exists
    if not os.path.exists(stl_path):
        return False
    
    # Check if the input result pkl file exists
    if not os.path.exists(result_pkl_path):
        return False
    
    # Load the mesh and do voxelization
    mesh = pv.read(stl_path)
    voxels = pv.voxelize(mesh, density=voxel_size)
    
    #voxels.plot(show_edges=True)

    # Load the result pkl file and get the tenon positions
    robot_result = pkl.load(open(result_pkl_path, 'rb'))
    input_stl_name_no_ext = stl_path.replace(".stl", "").split("/")[-1]
    
    link_dict = robot_result.link_dict[input_stl_name_no_ext]

    # print("Link name: ", input_stl_name_no_ext)
    # print("Link Tenon Positions: ", link_dict.tenon_pos)

    # Iterate through the tenon positions and check if a voxel is close to the tenon position
    distance_threshold = voxel_size * 1.732 * 2 # Use 2 as a safety factor to eliminate the voxel resolution error
    for tenon_pos_vec in link_dict.tenon_pos:
        tenon_pos = np.array(tenon_pos_vec)[:3]
        have_close_voxel = False
        for i in range(voxels.n_points):
            voxel_pos = voxels.points[i]
            distance = np.linalg.norm(voxel_pos - tenon_pos)
            if distance < distance_threshold:
                have_close_voxel = True
                break
        
        if not have_close_voxel:
            print("Destruction detected at tenon position: ", tenon_pos)
            # plot the voxels and the tenon position with different colors
            if plotting:
                p = pv.Plotter()
                p.add_mesh(voxels, color='red')
                p.add_mesh(pv.Sphere(center=tenon_pos, radius=distance_threshold), color='blue')
                p.show()    
            return False
        else:
            continue

    #print("No destruction detected.")
    return True
    

'''
Check if the mesh is destroyed in the interference removal for all the stl files in the urdf folder
@param folder_path: The path to the urdf folder
@param result_pkl_path: The path to the result pkl file
@param voxel_size: The voxel size used in the voxelization. Unit: m
'''
def destruction_check_urdf_folder(folder_path, result_pkl_path, voxel_size=0.005, plotting=True):
    # Find all the stl files in the urdf folder
    stl_files = [f for f in os.listdir(folder_path) if f.endswith('.stl')]
    print("Destruction Checker Found ", len(stl_files), " stl files in the folder: ", folder_path)
    print("Using the result pkl file: ", result_pkl_path)

    for stl_file in stl_files:
        stl_path = folder_path + "/" + stl_file
        if not destruction_check(stl_path, result_pkl_path, voxel_size, plotting):
            print("Destruction detected in the stl file: ", stl_file)
            return False
    
    print("No destruction detected in all the stl files.")

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if the mesh is destroyed in the interference removal')
    #parser.add_argument('--stl_path', type=str, default='/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241009-203842_flaw/FR_UP.stl', help='The path to the STL file. Unit: m')
    #parser.add_argument('--urdf_folder', type=str, default='/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241010-134328_good', help='The path to the urdf folder')
    #parser.add_argument('--result_pkl_path', type=str, default='/home/clarence/git/anything2robot/anything2robot/auto_design/results/gold_lynel20241010-134332_robot_result.pkl', help='The path to the result pkl file')
    
    parser.add_argument('--urdf_folder', type=str, default='/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241016-205927', help='The urdf folder path')
    parser.add_argument('--result_pkl_path', type=str, default='/home/clarence/git/anything2robot/anything2robot/auto_design/results/gold_lynel20241016-205928_robot_result.pkl', help='The pkl file path')
    
    parser.add_argument('--voxel_size', type=float, default=0.01, help='The voxel size used in the voxelization. Unit: m')

    args = parser.parse_args()

    #destruction_check(args.stl_path, args.result_pkl_path, args.voxel_size)
    
    destruction_check_urdf_folder(args.urdf_folder, args.result_pkl_path, args.voxel_size)