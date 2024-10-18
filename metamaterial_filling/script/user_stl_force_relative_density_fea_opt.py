'''
@Author: Clarence
@Date: 2024-7-5
@Description: This script is used to do the FEA analysis for the input STL file with given forces and fixed nodes. Then output the result, inlcuding relative density, to a pkl file.
'''

import pickle as pkl
import numpy as np
import sys
import os
import argparse
import trimesh
import open3d as o3d

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir + '/metamaterial_filling/script')
sys.path.append(project_dir + '/auto_design/modules')
sys.path.append(project_dir + '/auto_design')

from interference_removal import RobotOptResult, LinkResult
from stl_relative_density_fea_opt import do_static_fea
from io_interface.fea_result_class import FEA_Opt_Result
from simple_force_calculator import calculate_forces_from_nodes_and_torques


# Function to calculate the Euclidean distance between two points
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Load the STL file
def load_stl(file_path):
    return trimesh.load_mesh(file_path)

# Find the vertex that is the farthest from a given point
def find_farthest_vertex(stl_mesh, point):
    vertices = stl_mesh.vertices

    print("Vertices: ", vertices)
    print("Point: ", point)

    distances = np.linalg.norm(vertices - point, axis=1)
    farthest_vertex_index = np.argmax(distances)
    return vertices[farthest_vertex_index], distances[farthest_vertex_index]

# Create an arrow for visualization
def create_arrow(origin, direction, length=1.0, radius=0.05, resolution=20, color=[1, 0, 0]):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius,
        cone_radius=2 * radius,
        cylinder_height=0.8 * length,
        cone_height=0.2 * length,
        resolution=resolution,
        cylinder_split=4,
        cone_split=1
    )
    arrow.paint_uniform_color(color)
    arrow.translate(origin)

    # Calculate the rotation matrix to align the arrow with the direction vector
    direction = np.array(direction)
    direction = direction / np.linalg.norm(direction)  # Normalize the direction vector
    z_axis = np.array([0, 0, 1])
    rotation_matrix = np.eye(3)
    if not np.allclose(direction, z_axis):
        v = np.cross(z_axis, direction)
        c = np.dot(z_axis, direction)
        k = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        rotation_matrix = np.eye(3) + k + k.dot(k) * (1 / (1 + c))

    arrow.rotate(rotation_matrix, center=origin)
    return arrow

# Create a sphere for visualization
def create_sphere(center, radius=1.0, color=[1, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(center)
    return sphere


def stl_force_relative_density_fea_opt(stl_path_input=None, robot_result_file=None, output_folder=None, max_iteration=None, max_allowd_stress=None, max_allowd_displacement=None, display_fea_result=None, display_force_result=None, mapdl_object=None):
    parser = argparse.ArgumentParser(description="Static FEA Analysis for STL with given forces and fixed nodes")
    
    parser.add_argument("--input_stl_path", type=str, default='/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241010-134328_good/BODY_UP.stl', help="Path to the mesh .stl file")
    parser.add_argument("--robot_result_file", type=str, default='/home/clarence/git/anything2robot/anything2robot/auto_design/results/gold_lynel20241010-134332_robot_result.pkl', help="Path to the robot result including applied torques")

    # parser.add_argument("--input_stl_path", type=str, default='/home/clarence/git/anything2robot/anything2robot/urdf/gold_lynel20241016-212518_bad_example/BODY_UP.stl', help="Path to the mesh .stl file")
    # parser.add_argument("--robot_result_file", type=str, default='/home/clarence/git/anything2robot/anything2robot/auto_design/results/gold_lynel20241016-212518_robot_result.pkl', help="Path to the robot result including applied torques")


    parser.add_argument('--unit', type=str, default='m', choices=['mm', 'm'], help='Unit of the model. Note FEA uses mm as the unit. If the unit is in meter, we will scale the model to mm.')
    parser.add_argument('--output_folder', type=str, default='data/output', help='Output folder path')

    parser.add_argument('--torque2force_num', type=int, default=4, help='Number of forces generated for each torque')

    #### Parameters for tetrahedralMeshing
    parser.add_argument('--mesh_desired_element_number', type=int, default=10000, help='Desired number of elements in the mesh')
    parser.add_argument('--mesh_surface_accuracy', type=float, default=0.5, help='Surface accuracy of the mesh. Range: (0, 1]')

    #### Parameters for FEA
    parser.add_argument('--material_young_modulus', type=float, default=2310.0, help='Young\'s modulus of the material. MPa. Default. PC Material')
    parser.add_argument('--material_poisson_ratio', type=float, default=0.35, help='Poisson\'s ratio of the material')
    
    # The following default young_modulus_curve for meta-material is from https://www.sciencedirect.com/science/article/pii/S2352431619302640
    parser.add_argument('--young_modulus_curve_points_x', type=float, nargs='+', default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0], help='X values of the young modulus curve. Relative density.')
    parser.add_argument('--young_modulus_curve_points_y', type=float, nargs='+', default=[0.5, 0.52, 0.54, 0.565, 0.595, 0.625, 1.0], help='Y values of the young modulus curve. metamaterial_structure_young_modulus/(Relative density * material_young_modulus)') 

    parser.add_argument("--closest_node_num_per_fixed", type=int, default=100, help="Number of closest nodes to the fixed_nodes to fix the nodes")
    parser.add_argument("--closest_node_num_per_force", type=int, default=20, help="Number of closest nodes to the forces_nodes to apply the forces")

    ### Display parameters
    parser.add_argument("--display_fea_result", type=bool, default=True, help="Display the models")
    parser.add_argument("--display_force_result", type=bool, default=True, help="Display the forces")

    ### Optimization parameters
    parser.add_argument("--max_allowd_stress", type=float, default=3, help="Maximum allowed von Mises stress. MPa")
    parser.add_argument("--max_allowd_displacement", type=float, default=5, help="Maximum allowed displacement. mm")
    parser.add_argument("--max_iteration", type=int, default=1, help="Maximum number of iterations")
    parser.add_argument("--initial_relative_density", type=float, default=0.2, help="Initial relative density of the metamaterial structure")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for the gradient descent")

    #### Forces and fixed nodes. NOTE: No need to set!! Will be calculated from the Pkl file. But the following shouldn't be commented out.
    parser.add_argument("--fixed_nodes", type=float, nargs='+', default=[[0, 0, -100]], help="List of nodes where the fixed end is. Will read from pkl file. The format is [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]. Unit: mm")
    parser.add_argument("--forces_nodes", type=float, nargs='+', default=[[50, 50, 150], [0, 0, 200]], help="List of nodes where the forces are applied. Will read from pkl file. The format is [node1_x, node1_y, node1_z, node2_x, node2_y, node2_z, ...]. Unit: mm")
    parser.add_argument("--forces", type=float, nargs='+', default=[[0, 100, 0], [0, 0, -100]], help="List of forces applied at the nodes. Will read from pkl file. The format is [F1_x, F1_y, F1_z, F2_x, F2_y, F2_z, ...]. Unit: N")
    
    args = parser.parse_args()


    # Use function input if provided
    if stl_path_input is not None:
        args.input_stl_path = stl_path_input
    if robot_result_file is not None:
        args.robot_result_file = robot_result_file
    if output_folder is not None:
        args.output_folder = output_folder
    if max_iteration is not None:
        args.max_iteration = max_iteration
    if max_allowd_stress is not None:
        args.max_allowd_stress = max_allowd_stress
    if max_allowd_displacement is not None:
        args.max_allowd_displacement = max_allowd_displacement
    if display_fea_result is not None:
        args.display_fea_result = display_fea_result
    if display_force_result is not None:
        args.display_force_result = display_force_result

    # Clean the output folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    else:
        os.system("rm -r " + args.output_folder + "/*")

    # Load robot result
    robot_result = pkl.load(open(args.robot_result_file, 'rb'))
    input_stl_name_no_ext = args.input_stl_path.replace(".stl", "").split("/")[-1]
    
    link_dict = robot_result.link_dict[input_stl_name_no_ext]

    print("Link name: ", input_stl_name_no_ext)
    print("Link tenons: ", link_dict.tenon_pos)
    print("Link Torques: ", link_dict.applied_torque)
    #print("Joint type: ", link_dict.joint_type)


    # For the links with only one node. We need to add a fixed end.
    use_farthest_vertex_as_fixed_node = False
    if len(link_dict.tenon_pos) < 2:
        print("At least two applied_torque positions are required for FEA analysis. The first applied_torque position is used as the fixed node. Will use the fartherst vertex as the fixed node.")
        # Load the STL file. Later in FEA the stl will be scaled to mm.
        stl_mesh = load_stl(args.input_stl_path)
        torque_node_point = link_dict.tenon_pos[0][:3]
        print("Torque node point: ", torque_node_point)
        farthest_vertex, distance = find_farthest_vertex(stl_mesh, torque_node_point)
        print("Farthest vertex: ", farthest_vertex)

        # Use the point between the farthest vertex and the torque node as the fixed node
        fixed_node = farthest_vertex * 0.8 + torque_node_point * 0.2

        # Add the farthest vertex as the first element
        link_dict.tenon_pos.insert(0, np.array([fixed_node[0], fixed_node[1], fixed_node[2], 0, 0, 0]))
        link_dict.applied_torque.insert(0, np.array([fixed_node[0], fixed_node[1], fixed_node[2], 0, 0, 0]))
        use_farthest_vertex_as_fixed_node = True

        print("Link Torques: ", link_dict.applied_torque)


    if args.unit == 'm':
        link_dict.tenon_pos = np.array(link_dict.tenon_pos) * 1000 #mm
        link_dict.applied_torque = np.array(link_dict.applied_torque) * 1000 #mm, Nmm


    forces = calculate_forces_from_nodes_and_torques(link_dict.tenon_pos, link_dict.applied_torque)
    # Suppose forces direction is +z, to resist the gravity
    forces_vector = []
    for force in forces:
        forces_vector.append([0, 0, force])

    force_nodes_pos_list = []
    forces_list = []
    original_torque_node_list = []
    rotation_vector_list = []
    motor_radius = 30 # mm

    # Get the fixed nodes. Use the farthest point or find the index of the fixed node in the link_dict.applied_torque
    fixed_node_index = 0
    biggest_torque = 0
    if use_farthest_vertex_as_fixed_node:
        fixed_node_index = 0
        fixed_nodes = [[link_dict.tenon_pos[0][0], link_dict.tenon_pos[0][1], link_dict.tenon_pos[0][2]]]
    else:
        for i in range(len(link_dict.applied_torque)):
            if np.linalg.norm(link_dict.applied_torque[i][3:]) > biggest_torque:
                biggest_torque = np.linalg.norm(link_dict.applied_torque[i][3:])
                fixed_nodes = [[link_dict.tenon_pos[i][0], link_dict.tenon_pos[i][1], link_dict.tenon_pos[i][2]]]
                fixed_node_index = i

    non_fixed_nodes_indices = [i for i in range(len(link_dict.applied_torque)) if i != fixed_node_index]

    
    print("Fixed node: ", fixed_nodes)
    print("Non fixed nodes: ", non_fixed_nodes_indices)


    # Add forces
    for i in non_fixed_nodes_indices:
        force_node = [link_dict.tenon_pos[i][0], link_dict.tenon_pos[i][1], link_dict.tenon_pos[i][2]]
        force_nodes_pos_list.append(force_node)
        forces_list.append(forces_vector[i])

    
    # Get equivalent forces from the applied torques and add them
    for i in non_fixed_nodes_indices:
        # Set the force node and force vector. Start from the second tenon_pos position. The first applied_torque position is the fixed node
        original_torque_node_vector = [link_dict.tenon_pos[i][0], link_dict.tenon_pos[i][1], link_dict.tenon_pos[i][2]]
        original_torque_vector = [link_dict.applied_torque[i][3], link_dict.applied_torque[i][4], link_dict.applied_torque[i][5]]
        original_torque_length = np.linalg.norm(original_torque_vector)
        each_force_length = original_torque_length / motor_radius / args.torque2force_num

        original_torque_node_list.append(original_torque_node_vector)
        rotation_vector_list.append(original_torque_vector / np.linalg.norm(original_torque_vector))

        original_torque_node_vector_unit = original_torque_node_vector / np.linalg.norm(original_torque_node_vector)

        rotation_vector = original_torque_vector / np.linalg.norm(original_torque_vector)

        # Get a unit cross vector of the original_torque_vector and original_torque_node_vector
        cross_vector_unit = np.cross(rotation_vector, original_torque_node_vector_unit)
        cross_vector_unit = cross_vector_unit / np.linalg.norm(cross_vector_unit) # Not necessary

        force_node = original_torque_node_vector + cross_vector_unit * motor_radius
        force_vector_unit = np.cross(rotation_vector, cross_vector_unit)
        force_vector_unit = force_vector_unit / np.linalg.norm(force_vector_unit)

        force_nodes_pos_list.append(force_node)
        forces_list.append(force_vector_unit * each_force_length)
        

        # Get the rest of the forces
        rotate_angle_step = 2 * np.pi / args.torque2force_num
        force_node_to_rotation_center = force_node - original_torque_node_vector

        for j in range(1, args.torque2force_num):
            rotation_angle = rotate_angle_step * j
            # Rotate force_node_to_rotation_center around rotation_vector by rotation_angle
            rotation_matrix = np.array([[np.cos(rotation_angle) + rotation_vector[0]**2 * (1 - np.cos(rotation_angle)), rotation_vector[0] * rotation_vector[1] * (1 - np.cos(rotation_angle)) - rotation_vector[2] * np.sin(rotation_angle), rotation_vector[0] * rotation_vector[2] * (1 - np.cos(rotation_angle)) + rotation_vector[1] * np.sin(rotation_angle)],
                                         [rotation_vector[1] * rotation_vector[0] * (1 - np.cos(rotation_angle)) + rotation_vector[2] * np.sin(rotation_angle), np.cos(rotation_angle) + rotation_vector[1]**2 * (1 - np.cos(rotation_angle)), rotation_vector[1] * rotation_vector[2] * (1 - np.cos(rotation_angle)) - rotation_vector[0] * np.sin(rotation_angle)],
                                         [rotation_vector[2] * rotation_vector[0] * (1 - np.cos(rotation_angle)) - rotation_vector[1] * np.sin(rotation_angle), rotation_vector[2] * rotation_vector[1] * (1 - np.cos(rotation_angle)) + rotation_vector[0] * np.sin(rotation_angle), np.cos(rotation_angle) + rotation_vector[2]**2 * (1 - np.cos(rotation_angle))]])
            
            force_node_rotated = np.dot(rotation_matrix, force_node_to_rotation_center) + original_torque_node_vector
            force_nodes_pos_list.append(force_node_rotated)

            force_vector_unit_rotated = np.dot(rotation_matrix, force_vector_unit)
            forces_list.append(force_vector_unit_rotated * each_force_length)
        

    # VISUALIZATION
    if args.display_force_result:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        arrow = [None] * len(forces_list)
        for i in range(len(force_nodes_pos_list)):
            arrow[i] = create_arrow(force_nodes_pos_list[i], forces_list[i], length=np.linalg.norm(forces_list[i]), radius=0.5, resolution=20, color=[1, 0, 0])
            vis.add_geometry(arrow[i])

            print("Force node: ", force_nodes_pos_list[i])
            print("Force: ", forces_list[i])
        

        for i in range(len(original_torque_node_list)):
            arrow_torque = create_arrow(original_torque_node_list[i], rotation_vector_list[i], length=20, radius=0.5, resolution=20, color=[0, 1, 0])
            vis.add_geometry(arrow_torque)

        # Add the fixed nodes
        for i in range(len(fixed_nodes)):
            sphere = create_sphere(fixed_nodes[i], radius=10, color=[0, 0, 1])
            vis.add_geometry(sphere)

        vis.run()
        vis.destroy_window()
    
    # Do FEA analysis
    args.fixed_nodes = fixed_nodes
    args.forces_nodes = force_nodes_pos_list
    args.forces = forces_list

    success_flag, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes = do_static_fea(args, mapdl_object)

    # Store the FEA result
    fea_result = FEA_Opt_Result(input_stl_name_no_ext)
    
    nodes_seq_stress_exceeded = np.where(von_mises > args.max_allowd_stress)[0]

    fea_result.set_result(success_flag, best_relative_density, young_modulus, von_mises, displacement_magnitude, nodes, args.max_allowd_stress, args.max_allowd_displacement, nodes_seq_stress_exceeded)
            
    # Serialize the object to a pkl file
    pkl_file_path = os.path.dirname(os.path.abspath(__file__)) + "/../" + args.output_folder + "/" + input_stl_name_no_ext + "_fea_result.pkl"
    #pkl_file_path = os.path.join(args.output_folder, input_stl_name_no_ext + "_fea_result.pkl")
    
    with open(pkl_file_path, 'wb') as f:
        pkl.dump(fea_result, f)
    
    print("FEA result is saved to ", pkl_file_path)

    return success_flag, best_relative_density


if __name__ == "__main__":
    stl_force_relative_density_fea_opt(stl_path_input='/media/clarence/Clarence/anything2robot/urdf/gold_lynel20241018-110051/BODY_UP.stl', robot_result_file='/media/clarence/Clarence/anything2robot/auto_design/results/gold_lynel20241018-110054_robot_result.pkl')
