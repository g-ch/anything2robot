'''
@Author: Clarence
@Date: 2024-7-1
@Description: This script is used to do the FEA optimization for the metamaterial structure. The script will do meshing and FEA optimization to get the best relative density for the metamaterial structure.
'''

import os
import argparse
import subprocess
from format_transform.stl_to_off import stlToOff
from format_transform.vtu_to_ansys_msh import write_msh_file
from pyansys_fea.mapdl_msh_analysis import MapdlFea
import time
import trimesh

'''
@breif: Do meshing using tetrahedralMeshing
@param: stl_file: str: Path to the input stl file
@param: output_folder: str: Path to the output folder
@param: cmake_build_dir: str: Path to the cmake build directory to execute the tetrahedralMeshing
@param: mesh_desired_element_number: int: Desired number of elements in the mesh
@param: mesh_surface_accuracy: float: Surface accuracy of the mesh. Range: (0, 1]
@return: str: Path to the output msh file
'''
def meshing(stl_file: str, output_folder: str, cmake_build_dir: str, mesh_desired_element_number: int = 10000, mesh_surface_accuracy: float = 0.5):
    stl_file_name = os.path.basename(stl_file)
    off_file_name = stl_file_name.replace('.stl', '.off')
    mesh_vtu_file_name = stl_file_name.replace('.stl', '.vtu')
    mesh_msh_file_name = stl_file_name.replace('.stl', '.msh')

    # Convert the stl file to off file
    off_file_path = os.path.join(output_folder, off_file_name)
    off_file_path = cmake_build_dir + "/../" + off_file_path
    vtu_file_path = cmake_build_dir + "/../" + os.path.join(output_folder, mesh_vtu_file_name)
    msh_file_path = cmake_build_dir + "/../" + os.path.join(output_folder, mesh_msh_file_name)

    stlToOff(stl_file, off_file_path)

    print(f'STL file path: {stl_file}')
    print(f'Off file path: {off_file_path}')
    print(f'VTU file path: {vtu_file_path}')


    # Do Meshing using tetrahedralMeshing
    time.sleep(1)
    print(f'Meshing the model using tetrahedralMeshing...')
    subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./tetrahedralMeshing {off_file_path} {vtu_file_path} {mesh_desired_element_number} {mesh_surface_accuracy}'])  # Add "; exec bash" for keeping the terminal open
    
    # Check if the vtu file exists. If not wait for 1 second and check again.
    print(f'Waiting for the mesh vtu file to be generated in a seperate thread. Do not close the terminal!!!!!')
    while not os.path.exists(vtu_file_path):
        time.sleep(1)

    # Wait for 3 seconds to make sure the vtu file is generated. Big models may take longer time.
    time.sleep(5)

    # Convert the vtu file to msh file
    print(f'Converting the vtu file to msh file...')
    #write_msh_file(os.path.join(output_folder, mesh_vtu_file_name), os.path.join(output_folder, mesh_msh_file_name))
    write_msh_file(vtu_file_path, msh_file_path)

    # Wait for 3 seconds to make sure the msh file is generated. Big models may take longer time.
    time.sleep(5)

    print(f'Meshing completed! The msh file is saved at {msh_file_path}')
    return msh_file_path



'''
@breif: Get the equivalent young modulus for the metamaterial structure using linear interpolation
@param: material_young_modulus: float: Young's modulus of the material. MPa
@param: relative_density: float: Relative density of the metamaterial structure
@param: young_modulus_curve_points_x: list: X values of the young modulus curve. Relative density.
@param: young_modulus_curve_points_y: list: Y values of the young modulus curve. metamaterial_structure_young_modulus/(Relative density * material_young_modulus)
@return: float: Equivalent young modulus of the metamaterial structure
'''
def get_equivalent_young_modulus(material_young_modulus, relative_density, young_modulus_curve_points_x, young_modulus_curve_points_y):
    for i in range(len(young_modulus_curve_points_x) - 1):
        if relative_density >= young_modulus_curve_points_x[i] and relative_density <= young_modulus_curve_points_x[i+1]:
            # Linear interpolation
            interploation_value = young_modulus_curve_points_y[i] + (young_modulus_curve_points_y[i+1] - young_modulus_curve_points_y[i]) * (relative_density - young_modulus_curve_points_x[i]) / (young_modulus_curve_points_x[i+1] - young_modulus_curve_points_x[i])
            result = interploation_value * material_young_modulus * relative_density
            if result < 10:  # Set a lower limit for the young modulus in case of too small value
                return 10
            else:
                return interploation_value * material_young_modulus * relative_density
    
'''
@breif: Get the equivalent stress for the metamaterial structure considering the six fold plate structure
@param: stress: float: Von Mises stress of the metamaterial structure
@param: relative_density: float: Relative density of the metamaterial structure
'''
def get_equivalent_stress_micro_structure(stress, relative_density):
    # Consider 6 fold plate structure
    if relative_density == 0:
        return 10e12 # Very large number
    elif relative_density == 1:
        return stress
    else:
        return stress / relative_density * 2.63


'''
@breif: Main function to do the FEA optimization. The function will do meshing and FEA optimization to get the best relative density for the metamaterial structure.
        We use CGAL to do the meshing and pyansys to do the FEA analysis. The optimization is realized by computational gradient descent.
        
'''
def do_static_fea(args, mapdl_object=None):
    #################  Run checking  ########################
    # Check if the input STL file exists
    if not os.path.exists(args.input_stl_path):
        raise FileNotFoundError(f'Input STL file not found at {args.input_stl_path}')

    # Check if the output folder exists. If not, create it
    output_folder_global_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', args.output_folder)
    if not os.path.exists(output_folder_global_path):
        os.makedirs(output_folder_global_path)

    # Clear the output folder
    os.system(f'rm -rf {output_folder_global_path}/*')
    
    # Check if young_modulus_curve_points_x and young_modulus_curve_points_y have the same length > 2
    if len(args.young_modulus_curve_points_x) != len(args.young_modulus_curve_points_y):
        raise ValueError('The length of young_modulus_curve_points_x and young_modulus_curve_points_y must be the same')
    
    if len(args.young_modulus_curve_points_x) < 2:
        raise ValueError('The length of young_modulus_curve_points_x must be at least 2')
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    cmake_build_dir = os.path.join(current_dir, '../build')
    
    # If the unit is in meter, convert the unit to mm
    stl_file_name = os.path.basename(args.input_stl_path)
    scaled_file_name = stl_file_name.replace('.stl', '_scaled.stl')

    #ori_stl_file_path = current_dir + "/../" + args.input_stl_path

    ori_stl_file_path = args.input_stl_path
    scaled_file_path = current_dir + "/../" + args.output_folder + "/" + scaled_file_name

    # print(f'Original STL file path: {ori_stl_file_path}')
    # print(f'Scaled STL file path: {scaled_file_path}')

    #################  Scale if the mesh is in m rather than mm  ########################
    if args.unit == 'm':
        # Use gnome-terminal to run the command
        # print(f'Scaling the model to mm... ori_stl_file_path: {ori_stl_file_path}, scaled_file_path: {scaled_file_path}')
        # subprocess.run(['gnome-terminal', '--', 'bash', '-c', f'cd {cmake_build_dir}; ./scaleMesh {ori_stl_file_path} {scaled_file_path} 1000'])
        print(f'Scaling the model to mm...')

        # Use triMesh to scale the model
        mesh = trimesh.load(ori_stl_file_path)
        mesh.apply_scale(1000)
        mesh.export(scaled_file_path)

    else:
        print(f'Copying the model to the output folder...')
        os.system(f'cp {args.input_stl_path} {scaled_file_path}')
    
    # Check if the scaled file exists. If not wait for 1 second and check again.
    while not os.path.exists(scaled_file_path):
        time.sleep(1)

    # Wait for 3 seconds to make sure the scaled file is generated. Big models may take longer time.
    time.sleep(3)
    
    #################  Do meshing  ########################
    # Do meshing
    print(f'Meshing the model...')
    mesh_file_path = meshing(scaled_file_path, args.output_folder, cmake_build_dir, args.mesh_desired_element_number, args.mesh_surface_accuracy)
    
    # mesh_file_path = current_dir + "/../" + mesh_file_path


    #################  Do FEA optimization using searching by Computational Gradient Descent  ########################
    # Do FEA optimization
    mesh_file_path_no_ext = mesh_file_path.replace('.msh', '')
    print(f'Mesh file path: {mesh_file_path_no_ext}')

    recorded_relative_density = []
    recorded_von_mises = []
    recorded_displacement_magnitude = []

    # Check if using fully filled structure (relative_density=1) can meet the target
    relative_density = 1.0

    mapdl_created_in_this_function = False  # Flag to check if the mapdl object is created in this function
    if mapdl_object is None:
        mapdl_object = MapdlFea() # Create the mapdl object if not created
        mapdl_created_in_this_function = True

    max_stress, max_displacement,von_mises, displacement_magnitude, nodes  = mapdl_object.static_fea_analysis(msh_file=mesh_file_path_no_ext, elastic=args.material_young_modulus, poisson_ratio=args.material_poisson_ratio, fixed_nodes=args.fixed_nodes, closest_node_num_per_fixed=args.closest_node_num_per_fixed, forces_nodes=args.forces_nodes, forces=args.forces, closest_node_num_per_force=args.closest_node_num_per_force, display=args.display_fea_result)
    max_stress = get_equivalent_stress_micro_structure(max_stress, relative_density) # Get the equivalent stress for the metamaterial structure considering the six fold plate structure

    # Record
    recorded_relative_density.append(relative_density)
    recorded_von_mises.append(max_stress)
    recorded_displacement_magnitude.append(max_displacement)
    
    # Calculate the stress and displacement to the target to get gradient
    stress_to_allowed_value =  max_stress - args.max_allowed_stress_material
    displacement_to_allowed_value = max_displacement- args.max_allowed_displacement

    if stress_to_allowed_value > 0 or displacement_to_allowed_value > 0:
        print(f'The metamaterial cannot be used with the given conditions. The best relative density is 1')
        print(f'The best von Mises stress and displacement are:')
        print(f"Max von Mises stress: {max_stress}")
        print(f"Max displacement: {max_displacement}")
        return False, relative_density, args.material_young_modulus, von_mises, displacement_magnitude, nodes, recorded_relative_density, recorded_von_mises, recorded_displacement_magnitude

    # Check if the check_only flag is set. If yes, return the results without optimization
    if args.check_only:
        return True, relative_density, args.material_young_modulus, von_mises, displacement_magnitude, nodes, recorded_relative_density, recorded_von_mises, recorded_displacement_magnitude

    # Set the initial best values as the value of relative_density=1
    best_relative_density = relative_density
    stress_of_best_relative_density = max_stress
    displacement_of_best_relative_density = max_displacement

    # Start the optimization. Use user defined initial relative_density
    relative_density = args.initial_relative_density
    young_modulus = get_equivalent_young_modulus(args.material_young_modulus, relative_density, args.young_modulus_curve_points_x, args.young_modulus_curve_points_y)
    print(f'Equivalant young modulus: {young_modulus}')

    max_stress, max_displacement, von_mises, displacement_magnitude, nodes  = mapdl_object.static_fea_analysis(msh_file=mesh_file_path_no_ext, elastic=young_modulus, poisson_ratio=args.material_poisson_ratio, fixed_nodes=args.fixed_nodes, closest_node_num_per_fixed=args.closest_node_num_per_fixed, forces_nodes=args.forces_nodes, forces=args.forces, closest_node_num_per_force=args.closest_node_num_per_force, display=args.display_fea_result)
    print(f'density: {relative_density}, stress before correction: {max_stress}')
    max_stress = get_equivalent_stress_micro_structure(max_stress, relative_density) # Correct from macro structure to micro structure
    print(f'density: {relative_density}, stress after correction: {max_stress}')
    
    # Record
    recorded_relative_density.append(relative_density)
    recorded_von_mises.append(max_stress)
    recorded_displacement_magnitude.append(max_displacement)

    stress_to_allowed_value =  max_stress - args.max_allowed_stress_material
    displacement_to_allowed_value = max_displacement- args.max_allowed_displacement 

    print(f"The maximum von Mises stress and displacement of relative_density={relative_density} are:")
    print(f"Max von Mises stress: {max_stress}")
    print(f"Max displacement: {max_displacement}")
    print(f"Stress to target: {stress_to_allowed_value}")
    print(f"Displacement to target: {displacement_to_allowed_value}")

    # For the first iteration, we will decrease the relative_density by 0.1
    relative_density_new = relative_density - 0.1

    for i in range(args.max_iteration):
        print(f'Iteration {i+1}:')
        
        young_modulus = get_equivalent_young_modulus(args.material_young_modulus, relative_density_new, args.young_modulus_curve_points_x, args.young_modulus_curve_points_y)        
        max_stress_new, max_displacement_new, von_mises, displacement_magnitude, nodes  = mapdl_object.static_fea_analysis(msh_file=mesh_file_path_no_ext, elastic=young_modulus, poisson_ratio=args.material_poisson_ratio, fixed_nodes=args.fixed_nodes, closest_node_num_per_fixed=args.closest_node_num_per_fixed, forces_nodes=args.forces_nodes, forces=args.forces, closest_node_num_per_force=args.closest_node_num_per_force, display=args.display_fea_result)
        
        if max_stress_new > 1e10 or max_stress_new < 1e-5: # not feasible. Something wrong with the model. Terminate the optimization
            print(f'Optimization terminated. The model is not feasible.')
            break

        max_stress_new = get_equivalent_stress_micro_structure(max_stress_new, relative_density_new) # Correct from macro structure to micro structure

        # Record
        recorded_relative_density.append(relative_density)
        recorded_von_mises.append(max_stress)
        recorded_displacement_magnitude.append(max_displacement)

        stress_to_allowed_value_new = max_stress_new - args.max_allowed_stress_material
        displacement_to_allowed_value_new = max_displacement_new - args.max_allowed_displacement

        print(f"The maximum von Mises stress and displacement of relative_density={relative_density_new} are:")
        print(f"Max von Mises stress: {max_stress_new}")
        print(f"Max displacement: {max_displacement_new}")
        print(f"Stress to target: {stress_to_allowed_value_new}")
        print(f"Displacement to target: {displacement_to_allowed_value_new}")

        # Check if the stress and displacement are within the target
        if stress_to_allowed_value_new < 0 and displacement_to_allowed_value_new < 0:
            if relative_density_new < best_relative_density:
                # If the stress and displacement are within the target, we choose the one with the smallest relative_density as the best solution
                best_relative_density = relative_density_new
                stress_of_best_relative_density = max_stress_new
                displacement_of_best_relative_density = max_displacement_new

        # Update the relative_density by gradient descent
        gradient_stress = (stress_to_allowed_value_new - stress_to_allowed_value) / (relative_density_new - relative_density)
        gradient_displacement = (displacement_to_allowed_value_new - displacement_to_allowed_value) / (relative_density_new - relative_density)

        # Choose the gradient with the largest absolute value
        if abs(gradient_stress) > abs(gradient_displacement):
            gradient = gradient_stress
            distance_to_target = stress_to_allowed_value_new
        else:
            gradient = gradient_displacement
            distance_to_target = displacement_to_allowed_value_new
        
        # Store the current values for the next iteration
        relative_density = relative_density_new
        stress_to_allowed_value = stress_to_allowed_value_new
        displacement_to_allowed_value = displacement_to_allowed_value_new

        # Update the relative_density
        if distance_to_target > 0:
            relative_density_new = relative_density - args.learning_rate * gradient
        else:
            relative_density_new = relative_density + args.learning_rate * gradient

        # Check if the relative_density is in the range [0.05, 1]. Smaller than 0.05 is mostly unprintable and larger than 1 is fully filled.
        if relative_density_new < 0.05:
            relative_density_new = 0.05
        elif relative_density_new > 1:
            relative_density_new = 1

        time.sleep(3) # Wait for 2 seconds to make sure the FEA analysis is completed
    

    print(f'Optimization completed! The best relative density is {best_relative_density}')
    print(f'The best von Mises stress and displacement are:')
    print(f"Max von Mises stress: {stress_of_best_relative_density}")
    print(f"Max displacement: {displacement_of_best_relative_density}")
    young_modulus = get_equivalent_young_modulus(args.material_young_modulus, best_relative_density, args.young_modulus_curve_points_x, args.young_modulus_curve_points_y)
    print(f'The equivalent young modulus is {young_modulus}')

    # Check if the mapdl object is created in this function. If yes, shutdown the mapdl object
    if mapdl_created_in_this_function:
        mapdl_object.shutdown()

    return True, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes, recorded_relative_density, recorded_von_mises, recorded_displacement_magnitude


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate the final model and export to an STL file')

    #### Parameters for input and output
    #### NOTE: The input STL file must be in the relative path from root folder of this package
    parser.add_argument('--input_stl_path', type=str, default='data/FL.stl', help='Input STL file path')
    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. Note FEA uses mm as the unit. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--output_folder', type=str, default='data/output', help='Output folder path')

    parser.add_argument('--check_only', type=bool, default=False, help='Check only. If true, only solid body is considered.')

    #### Parameters for tetrahedralMeshing
    parser.add_argument('--mesh_desired_element_number', type=int, default=10000, help='Desired number of elements in the mesh')
    parser.add_argument('--mesh_surface_accuracy', type=float, default=0.5, help='Surface accuracy of the mesh. Range: (0, 1]')

    #### Parameters for FEA
    parser.add_argument('--material_young_modulus', type=float, default=3100.0, help='Young\'s modulus of the material. MPa')
    parser.add_argument('--material_poisson_ratio', type=float, default=0.35, help='Poisson\'s ratio of the material')
    # The default young_modulus_curve for meta-material is from https://www.sciencedirect.com/science/article/pii/S2352431619302640
    parser.add_argument('--young_modulus_curve_points_x', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], help='X values of the young modulus curve. Relative density.')
    parser.add_argument('--young_modulus_curve_points_y', type=float, nargs='+', default=[0.5, 0.52, 0.54, 0.565, 0.595, 0.625, 1.0], help='Y values of the young modulus curve. metamaterial_structure_young_modulus/(Relative density * material_young_modulus)') 
    
    parser.add_argument("--fixed_nodes", type=float, nargs='+', default=[[0, 0, -100]], help="List of nodes where the fixed end is. The format is [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]. Unit: mm")
    parser.add_argument("--closest_node_num_per_fixed", type=int, default=20, help="Number of closest nodes to the fixed_nodes to fix the nodes")

    parser.add_argument("--forces_nodes", type=float, nargs='+', default=[[50, 50, 150], [0, 0, 200]], help="List of nodes where the forces are applied. The format is [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]. Unit: mm")
    parser.add_argument("--forces", type=float, nargs='+', default=[[0, 100, 0], [0, 0, -100]], help="List of forces applied at the nodes. The format is [F1_x, F1_y, F1_z, F2_x, F2_y, F2_z, ...]. Unit: N")
    parser.add_argument("--closest_node_num_per_force", type=int, default=1, help="Number of closest nodes to the forces_nodes to apply the forces")
    parser.add_argument("--display_fea_result", type=bool, default=False, help="Display the models")

    ### Optimization parameters
    parser.add_argument("--max_allowed_stress_material", type=float, default=76, help="Maximum allowed von Mises stress. MPa")
    parser.add_argument("--max_allowed_displacement", type=float, default=2, help="Maximum allowed displacement. mm")
    parser.add_argument("--max_iteration", type=int, default=5, help="Maximum number of iterations")
    parser.add_argument("--initial_relative_density", type=float, default=0.2, help="Initial relative density of the metamaterial structure")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the gradient descent")
    args = parser.parse_args()

    success_flag, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes = do_static_fea(args)
    