'''
This script is used to view the mesh using trimesh and pyvista.
'''

import trimesh
import pyvista as pv    
import os
import readchar

def view_mesh(mesh_path):
    '''
    This function is used to view the mesh using trimesh and pyvista.
    '''
    mesh = trimesh.load(mesh_path)
    pv.plot(mesh)


if __name__ == "__main__":
    mesh_folder = "/media/clarence/Clarence/anything2robot_data/result"

    # Find all the subfolders in the mesh_folder. If one subfolder has subsubfolders named "result_roundN", where N is an integer, 
    # then view the mesh named BODY.stl in the "urdf folder" in the subsubfolder with the largest N.
    for folder in os.listdir(mesh_folder):
        if os.path.isdir(os.path.join(mesh_folder, folder)):
            max_round_num = 0
            for subfolder in os.listdir(os.path.join(mesh_folder, folder)):
                if os.path.isdir(os.path.join(mesh_folder, folder, subfolder)):
                    if "result_round" in subfolder:
                        #round_num = int(subfolder.split("_")[-1])
                        round_num = int(subfolder.split("_")[-1].replace("round", ""))
                        if round_num > max_round_num:
                            max_round_num = round_num

            # View the mesh
            round_num_str = f"result_round{max_round_num}"
            mesh_path = os.path.join(mesh_folder, folder, round_num_str, 'urdf', 'BODY.stl')

            # Check if the mesh file exists
            if not os.path.exists(mesh_path):
                print(f"Mesh file {mesh_path} does not exist")
                continue

            print(f"Viewing the mesh in {mesh_path}")
            view_mesh(mesh_path)

            # Wait for user input to continue
            input("Press Enter to continue... and press q or esc to quit")
            key = readchar.readkey()
            if key == 'q' or key == 'esc':
                break
