import numpy as np
import pickle as pkl
import argparse
import numpy as np
import os
import plotly
import pinocchio as pin
import open3d as o3d
import time
from urdf_generator import write_link, write_joint, write_material, voxel_grid_to_mesh, calculate_inertia_tensor
from mesh_decomp import Mesh_Decomp, Mesh_Group
from mesh_loader import Custom_Mesh_Loader
from motor_opt import Motor_Opt, get_bounds, Joint_Connect_Opt, is_points_in_cylinder
from itertools import product


apply_transform = lambda x, T: np.dot(T, np.hstack([x, np.ones((x.shape[0], 1))]).T).T[:, :3]
get_removed_list = lambda list, remove_value: [value for value in list if value != remove_value]

def get_tenon_idx(motor_result, motor_lib):
    
    motor_height = np.linalg.norm(motor_result[:3] - motor_result[3:6])
    motor_radius = motor_result[6]
    for i, motor in enumerate(motor_lib):
        if np.isclose(motor[0], motor_height, atol=0.1) and np.isclose(motor[1], motor_radius, atol=0.1):
            return i
    return -1


def expand_points(array):
    """
    Expand a given (n, 3) numpy array to include all integer points around each point in the array.
    
    Parameters:
    - array (np.ndarray): An (n, 3) numpy array of floats.
    
    Returns:
    - np.ndarray: An expanded (n+m, 3) array including surrounding integer points.
    """
    # Define the offsets to generate surrounding points
    offsets = np.array(list(product([-1, 0, 1], repeat=3)))  # Cartesian product to generate combinations
    
    # Initialize a list to hold all the new points
    new_points = []
    
    # Loop through each point in the original array
    for point in array:
        # Round the point to the nearest integer to find the central cube position
        center = np.round(point).astype(int)
        
        # Generate surrounding points by adding the offsets to the central position
        surrounding_points = center + offsets
        
        # Append the new points to the list
        new_points.append(surrounding_points)
    
    # Concatenate all points into a single array
    expanded_array = np.vstack(new_points)
    
    # Optionally, you might want to remove duplicates if exact duplicates are not desired
    expanded_array = np.unique(expanded_array, axis=0)

    return expanded_array

def plotly_3d_nparray(nparray):
    voxel_types = np.unique(nparray)
    colors = np.random.randint(0, 255, (2*len(voxel_types), 3))
    data = []
    for voxel_type in voxel_types:
        if voxel_type == 0:
            continue
        x, y, z = np.where(nparray == voxel_type)
        data.append(plotly.graph_objs.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=2, color='rgb({},{},{})'.format(*colors[voxel_type-1]))))
    fig = plotly.graph_objs.Figure(data=data)
    fig.show()


class LinkResult:
    def __init__(self):
        self.applied_force = []
        self.applied_torque = []
        self.tenon_pos = []
        self.tenon_type = []
        self.tenon_idx = []

    def add_force(self, force):
        self.applied_force.append(force)
    
    def add_torque(self, torque):
        self.applied_torque.append(torque)
    
    def add_tenon_pos(self, tenon_pos, tenon_type, tenon_idx):
        self.tenon_pos.append(tenon_pos)
        self.tenon_type.append(tenon_type)
        self.tenon_idx.append(tenon_idx)

class RobotOptResult:
    def __init__(self, interference_removal, urdf_dir, motor_lib):
        self.motor_results = interference_removal.motor_param_result
        self.link_tree = interference_removal.link_tree
        self.mesh_group = interference_removal.mesh_group
        self.father_link_dict = interference_removal.father_link_dict
        self.link_axis_dict = {}
        self.urdf_dir = urdf_dir
        self.args = interference_removal.args
        self.motor_lib = motor_lib
        self.foot_force = 40 # N
        self.tenon_height = 0 # cm
        self.link_dict = {}

        # 1. Add tenon position
        self.link_dict['BODY'] = LinkResult()
        cur_idx = 0
        queue = [self.link_tree]
        while queue:
            current_node = queue.pop(0)
            current_link = current_node.val
            for child_node in current_node.children:
                queue.append(child_node)
            # Skip BODY link
            if current_link.axis is None or np.linalg.norm(current_link.axis[1]) == 0:
                continue

            if len(current_link.axis) == 2:
                motor_direct = np.array(current_link.axis[1])
                motor_child_side = self.motor_results[cur_idx][3:6]
                motor_father_side = self.motor_results[cur_idx][:3]
                motor_idx = get_tenon_idx(self.motor_results[cur_idx], motor_lib)

                self.link_axis_dict[current_link.name+ '_joint'] = np.hstack(((motor_father_side + motor_child_side) / 200.0, motor_direct))

                self.link_dict[current_link.name] = LinkResult()
                tenon_pos = (motor_child_side + self.tenon_height * motor_direct) / 100.0
                tenon_direct = -motor_direct
                self.link_dict[current_link.name].add_tenon_pos(np.hstack((tenon_pos, tenon_direct)), 'child', motor_idx) 

                tenon_pos_father = (motor_father_side - self.tenon_height * motor_direct) / 100.0
                tenon_direct_father = motor_direct
                self.link_dict[self.father_link_dict[current_link.name]].add_tenon_pos(np.hstack((tenon_pos_father, tenon_direct_father)), 'father', motor_idx)
                cur_idx += 1
            
            elif len(current_link.axis) == 3:
                
                motor_direct1 = np.array(current_link.axis[1])
                motor_direct2 = np.array(current_link.axis[2])
                motor_pos1 = (self.motor_results[cur_idx][:3] + self.motor_results[cur_idx][3:6]) / 2
                motor_pos2 = (self.motor_results[cur_idx+1][:3] + self.motor_results[cur_idx+1][3:6]) / 2
                motor_child_side = self.motor_results[cur_idx][3:6]
                motor_father_side = self.motor_results[cur_idx+1][3:6]
                motor_idx1 = get_tenon_idx(self.motor_results[cur_idx], motor_lib)
                motor_idx2 = get_tenon_idx(self.motor_results[cur_idx+1], motor_lib)

                self.link_axis_dict[current_link.name + '_joint1'] = np.hstack((motor_pos2 / 100, motor_direct2))
                self.link_axis_dict[current_link.name + '_joint2'] = np.hstack((motor_pos1 / 100, motor_direct1))

                self.link_dict[current_link.name] = LinkResult()
                tenon_pos = (motor_child_side + self.tenon_height * motor_direct1) / 100.0
                tenon_direct = -motor_direct1
                self.link_dict[current_link.name].add_tenon_pos(np.hstack((tenon_pos, tenon_direct)), 'child', motor_idx1) 

                tenon_pos_father = (motor_father_side - self.tenon_height * motor_direct2) / 100.0
                tenon_direct_father = -motor_direct2
                self.link_dict[self.father_link_dict[current_link.name]].add_tenon_pos(np.hstack((tenon_pos_father, tenon_direct_father)), 'father', motor_idx2)
                cur_idx += 2
        
        #check the tenon position with mesh
        ## Visualize the stl file
        # def get_cross_prod_mat(pVec_Arr):
        #     # pVec_Arr shape (3)
        #     qCross_prod_mat = np.array([
        #         [0, -pVec_Arr[2], pVec_Arr[1]], 
        #         [pVec_Arr[2], 0, -pVec_Arr[0]],
        #         [-pVec_Arr[1], pVec_Arr[0], 0],
        #     ])
        #     return qCross_prod_mat
        # def caculate_align_mat(pVec_Arr):
        #     scale = np.linalg.norm(pVec_Arr)
        #     pVec_Arr = pVec_Arr/ scale
        #     # must ensure pVec_Arr is also a unit vec. 
        #     z_unit_Arr = np.array([0,0,1])
        #     z_mat = get_cross_prod_mat(z_unit_Arr)

        #     z_c_vec = np.matmul(z_mat, pVec_Arr)
        #     z_c_vec_mat = get_cross_prod_mat(z_c_vec)

        #     if np.dot(z_unit_Arr, pVec_Arr) == -1:
        #         qTrans_Mat = -np.eye(3, 3)
        #     elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        #         qTrans_Mat = np.eye(3, 3)
        #     else:
        #         qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
        #                                                     z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

        #     qTrans_Mat *= scale
        #     return qTrans_Mat
        # link_name = 'FL_UP'
        # mesh = o3d.io.read_triangle_mesh('./" + dir + "' + link_name + '.stl')
        # mesh.compute_vertex_normals()
        # tenon_positions = []
        # tenon_vector_o3d = []
        # for tenon_position_direcct in self.link_dict[link_name].tenon_pos:
        #     tenon_position = tenon_position_direcct[:3]
        #     tenon_direction = tenon_position_direcct[3:]
        #     arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.005, cone_radius=0.008, cylinder_height=0.04, cone_height=0.01)
        #     arrow.paint_uniform_color([1, 0, 0])

        #     rot_mat = caculate_align_mat(tenon_direction)
        #     arrow.rotate(rot_mat, center = (0,0,0))
        #     arrow.translate(tenon_position)
        #     tenon_vector_o3d.append(arrow)
        # arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.01, cone_radius=0.02, cylinder_height=0.05, cone_height=0.05)
        # o3d.visualization.draw_geometries([mesh] + tenon_vector_o3d)
        
        # 2. Add force and torque information
        pkg_dir = './..'
        model, collision_model, visual_model = pin.buildModelsFromUrdf(self.urdf_dir, pkg_dir)
        data = model.createData()
        max_torque = np.zeros((model.nv))
        def contact_link_func(link):
            for joint_name in link.joints.keys():
                if 'foot' in joint_name:
                    return True
            return False
        contact_nodes = self.link_tree.find_children(contact_link_func, None)
        contact_transformations = []
        contact_links = []
        for contact_node in contact_nodes:
            contact_links.append(contact_node.val.name+ '_joint')
            for joint_name, joint_pos in contact_node.val.joints.items():
                if 'foot' in joint_name:
                    contact_transformations.append(np.array(np.array(joint_pos) - contact_node.val.axis[0]) / 100.0)
                    break
        force = np.array([0, 0, self.foot_force])
        for i in range(10):
            q = pin.randomConfiguration(model)
            v = np.zeros((model.nv, 1))  # in rad/s 
            a = np.zeros((model.nv, 1))  # in rad/s²  
            fs_ext = [pin.Force(np.array([0,0,0,0,0,0])) for _ in range(len(model.joints))]

            for i, joint_name in enumerate(model.names):
                if joint_name in contact_links:
                    contact_id = contact_links.index(joint_name)
                    transformed_torque = np.cross(contact_transformations[contact_id], force)
                    fs_ext[i] = pin.Force(np.hstack((force, transformed_torque)))
            tau = pin.rnea(model, data, q, v, a, fs_ext)
            max_torque = np.maximum(max_torque, np.abs(tau))
        
        joint_torques = {}
        for joint_name, torque in zip(model.names, max_torque):
            if joint_name != 'universe':
                suffixes = ["_joint", "_joint1", "_joint2"]
                for suffix in suffixes:
                    if joint_name.endswith(suffix):
                        link_name = joint_name[:-len(suffix)]
                        break
                father_torque = self.link_axis_dict[joint_name]
                father_torque[3:] = father_torque[3:] * torque
                child_torque = father_torque.copy()
                child_torque[:3] = -child_torque[:3]

                if suffix == "_joint":
                    self.link_dict[self.father_link_dict[link_name]].add_torque(father_torque)
                    self.link_dict[link_name].add_torque(child_torque)
                elif suffix == "_joint1":
                    self.link_dict[self.father_link_dict[link_name]].add_torque(father_torque)
                    # self.link_dict[link_name].add_torque(child_torque)
                else:
                    # self.link_dict[self.father_link_dict[link_name]].add_torque(father_torque)
                    self.link_dict[link_name].add_torque(child_torque)
                
class InterferenceRemoval:
    def __init__(self, args, mesh_group : Mesh_Group, motor_param_result, link_tree, father_link_dict):
        self.args = args
        self.mesh_group = mesh_group
        self.motor_param_result = motor_param_result
        self.link_tree = link_tree
        self.father_link_dict = father_link_dict
        self.link_motor_dict = {}

        def contact_link_func(link):
            for joint_name in link.joints.keys():
                if 'foot' in joint_name:
                    return True
            return False
        self.contact_nodes = self.link_tree.find_children(contact_link_func, None)

        cur_idx = 0
        queue = [self.link_tree]
        while queue:
            current_node = queue.pop(0)
            current_link = current_node.val
            for child_node in current_node.children:
                queue.append(child_node)
            if current_link.axis is None or np.linalg.norm(current_link.axis[1]) == 0:
                continue
            motor_position = (self.motor_param_result[cur_idx][:3] + self.motor_param_result[cur_idx][3:6]) / 2
            motor_direct = np.array(current_link.axis[1])
            motor_radius = self.motor_param_result[cur_idx][6]
            self.link_motor_dict[current_link.name] = [(motor_position, motor_direct, motor_radius)]
            cur_idx += 1
            if len(current_link.axis) == 3:
                motor2_pos = (self.motor_param_result[cur_idx][:3] + self.motor_param_result[cur_idx][3:6]) / 2
                motor2_direct = np.array(current_link.axis[2])
                self.link_motor_dict[current_link.name].append((motor2_pos, motor2_direct, motor_radius))
                cur_idx += 1

    def set_joint_limit(self, joint_limits):
        self.joint_limits = joint_limits

    def rotate_around_axis(self, axis, theta, pos):

        def translation_matrix(vector):
            T = np.identity(4)
            T[:3, 3] = vector
            return T
        
        def generate_rotation_matrix(axis, theta):
            axis = np.asarray(axis)
            axis = axis / np.sqrt(np.dot(axis, axis))
            a = np.cos(theta / 2.0)
            b, c, d = -axis * np.sin(theta / 2.0)

            aa, bb, cc, dd = a*a, b*b, c*c, d*d
            bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
            return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
        
        # Step 1: Translate so that rotation axis passes through the origin
        T_inv = translation_matrix(-np.array(pos))

        # Step 2: Rotate around the axis
        R = generate_rotation_matrix(axis, theta)
        R_homo = np.identity(4)
        R_homo[:3, :3] = R

        # Step 3: Translate back
        T = translation_matrix(np.array(pos))

        # Combined transformation
        return T @ R_homo @ T_inv

    def remove_interference(self):
        queue = [self.link_tree]
        cur_idx = 0

        while queue:
            current_node = queue.pop(0)
            current_link = current_node.val

            for child_node in current_node.children:
                queue.append(child_node)
            if current_link.axis is None or np.linalg.norm(current_link.axis[1]) == 0:
                continue

            father_link = self.father_link_dict[current_link.name]
            father_link_value = self.mesh_group.link_value_dict[father_link]
            current_link_value = self.mesh_group.link_value_dict[current_link.name]
            rotation_num = 10

            # angle_resolution = abs(self.joint_limits[cur_idx, 0] - self.joint_limits[cur_idx, 1]) / rotation_num
            # negative_num = round(abs(self.joint_limits[cur_idx, 0]) / angle_resolution)
            # positive_num = round(abs(self.joint_limits[cur_idx, 1]) / angle_resolution)

            other_link_values = [value for value in self.mesh_group.link_value_dict.values() if value != father_link_value]

            # Get the info about current link
            motor_position = (self.motor_param_result[cur_idx][:3] + self.motor_param_result[cur_idx][3:6]) / 2
            motor_direct = np.array(current_link.axis[1])
            # motor_radius = self.motor_param_result[cur_idx][6]
            cur_idx += 1

            # Sample joint angles and implement the interference removal
            # for i in range(rotation_num):  #old: #
            #     if i < negative_num:
            #         joint_angle = -i * angle_resolution
            #         other_link_values = [value for value in self.mesh_group.link_value_dict.values() if value != father_link_value]
            #     else:
            #         joint_angle = (i - negative_num) * angle_resolution
            #         other_link_values = [value for value in self.mesh_group.link_value_dict.values() if value != father_link_value]

            for joint_angle in np.linspace(self.joint_limits[cur_idx, 0], self.joint_limits[cur_idx, 1], rotation_num):
                child_nodes = current_node.get_all_children()[0]
                transformed_links = [child.val for child in child_nodes]
                transformed_links.append(current_link)
                transformed_link_names = [link.name for link in transformed_links]
                
                transformed_indexs = np.vstack([self.mesh_group.get_voxels(link_name, get_index=True) for link_name in transformed_link_names])
                
                # The transformation is defined as rotation around the axis of the motor, rotating angle is the joint angle
                H_matrix = self.rotate_around_axis(motor_direct, joint_angle, self.mesh_group.position_to_index(motor_position.reshape(-1, 3)))

                new_indexs = expand_points(apply_transform(transformed_indexs, H_matrix))  
                new_indexs = np.clip(new_indexs, 0, self.mesh_group.voxel_data.shape[0]-1)

                values = self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]]  # Value Point type.
                non_removal = self.mesh_group.voxel_no_removal[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]]


                self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]] = np.where(np.logical_or(np.isin(values, other_link_values), non_removal),
                                                                                                            self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]], 
                                                                                                            0)

            # interference removal for the second motor
            if len(current_link.axis) == 3:

                motor_position = (self.motor_param_result[cur_idx][:3] + self.motor_param_result[cur_idx][3:6]) / 2
                motor_direct = np.array(current_link.axis[2]) 
                
                # Sample joint angles and implement the interference removal
                # for i in range(rotation_num):  #old: #
                #     if i < negative_num:
                #         joint_angle = -i * angle_resolution
                #         other_link_values = [value for value in self.mesh_group.link_value_dict.values() if value != father_link_value]
                #     else:
                #         joint_angle = (i - negative_num) * angle_resolution
                #         other_link_values = [value for value in self.mesh_group.link_value_dict.values() if value != father_link_value]
                    
                for joint_angle in np.linspace(self.joint_limits[cur_idx, 0], self.joint_limits[cur_idx, 1], rotation_num):
                    child_nodes = current_node.get_all_children()[0]
                    transformed_links = [child.val for child in child_nodes]
                    transformed_links.append(current_link)
                    transformed_link_names = [link.name for link in transformed_links]
                    
                    transformed_indexs = np.vstack([self.mesh_group.get_voxels(link_name, get_index=True) for link_name in transformed_link_names])


                    # The transformation is defined as rotation around the axis of the motor, rotating angle is the joint angle
                    H_matrix = self.rotate_around_axis(motor_direct, joint_angle, self.mesh_group.position_to_index(motor_position.reshape(-1, 3)))
                    
                    new_indexs = expand_points(apply_transform(transformed_indexs, H_matrix))
                    new_indexs = np.clip(new_indexs, 0, self.mesh_group.voxel_data.shape[0]-1)
                    values = self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]]
                    non_removal = self.mesh_group.voxel_no_removal[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]]
                    
                    self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]] = np.where(np.logical_or(np.isin(values, other_link_values), non_removal),
                                                                                                                self.mesh_group.voxel_data[new_indexs[:, 0], new_indexs[:, 1], new_indexs[:, 2]], 
                                                                                                                0)

                cur_idx += 1

        # Remove interference for motors
        for motor_param in self.motor_param_result:
            def condition_remove(pts):
                return is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 0, 0.5)
            self.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_group.link_value_dict.keys()), "Unoccupied"),
                                        target_group_name="Unoccupied",
                                        condition_func=condition_remove)

            

    def generate_urdf(self):
        """
        Generate the URDF file.
        """
        timestr = time.strftime("%Y%m%d-%H%M%S")
        dir = './urdf/' + self.args.model_name + timestr + '/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        urdf_file = open(dir + self.args.model_name + timestr + '.urdf', 'w+')
        package_name = "anything2robot"

        self.urdf_dir = dir + self.args.model_name + timestr + '.urdf'
        urdf_file.write('<?xml version="1.0"?>\n')
        urdf_file.write('<robot name="robot">\n')
        
        # Define materials
        materials = {
            "red": "1 0 0 1", "magenta": "0.2 0 0.2 1", "green": "0 1 0 1",
            "cyan": "0 1 1 1", "blue": "0 0 1 1", "orange": "1 0.5 0 1",
            "yellow": "1 1 0 1", "pink": "1 0 1 1", "grey": "0.8 0.8 0.8 1", "motor":"0.2 0.2 0.2 1"
        }
        for material_name, rgba in materials.items():
            write_material(urdf_file, material_name, rgba)

        cur_node = self.link_tree
        node_queue = [cur_node]

        self.ideal_mass = 0

        while node_queue:
            root_node = node_queue.pop(0)
            child_nodes = root_node.children
            for child_node in child_nodes:
                node_queue.append(child_node)
            
            if root_node.val.axis is None or np.linalg.norm(root_node.val.axis[1]) == 0:
                
                # Write BODY link
                voxel_grid_to_mesh(voxel_positions=self.mesh_group.get_voxels("BODY"), dir=dir + 'BODY.stl', voxel_size=self.args.voxel_size)
                link_visual = {
                    "origin": {"xyz": "0 0 0", "rpy": "0 0 0"},
                    "geometry": {"filename": "package://" + package_name + "/" + dir + "BODY.stl"},
                    "material": "grey"
                }
                link_collision = {
                    "origin": {"xyz": "0 0 0", "rpy": "0 0 0"},
                    "geometry": {"filename": "package://" + package_name + "/" + dir + "BODY.stl"}
                }
                per_voxel_mass = self.args.voxel_density * (self.args.voxel_size ** 3)
                part_mass = per_voxel_mass * self.mesh_group.get_voxels("BODY").shape[0]
                inetial_matrix, CoM = calculate_inertia_tensor(self.mesh_group.get_voxels("BODY") / 100.0, part_mass, np.eye(4))
                link_inertial = {
                    "origin": {"xyz": ' '.join(map(str, CoM)), "rpy": '0 0 0'},
                    "mass": str(self.mesh_group.get_voxels("BODY").shape[0] * self.args.voxel_density),
                    "inertia": {"ixx": inetial_matrix[0, 0], "iyy": inetial_matrix[1, 1], "izz": inetial_matrix[2, 2], "ixy": inetial_matrix[0, 1], "ixz": inetial_matrix[0, 2], "iyz": inetial_matrix[1, 2]}
                }
                write_link(urdf_file=urdf_file, link_name="BODY", visual=link_visual, collision=link_collision, inertial=link_inertial)
                
                continue

            # Write current link
            cur_link = root_node.val
            motor_pos, motor_direct, motor_radius = self.link_motor_dict[cur_link.name][-1]
            father_motor_pos = self.link_motor_dict[self.father_link_dict[cur_link.name]][0][0] if self.father_link_dict[cur_link.name] != "BODY" else np.zeros(3)

            link_dir = dir + cur_link.name + '' + '.stl'
            voxel_grid_to_mesh(voxel_positions=self.mesh_group.get_voxels(cur_link.name), dir=link_dir, voxel_size=self.args.voxel_size)
            rel_pos = (motor_pos - father_motor_pos) / 100.0 if cur_link.name != "BODY" else motor_pos / 100.0
            
            visual_pos = -np.array(motor_pos) / 100.0 if len(cur_link.axis) == 2 else -self.link_motor_dict[cur_link.name][0][0] / 100.0

            link_visual = {
                "origin": {"xyz": ' '.join(map(str, visual_pos)), "rpy": '0 0 0'},
                "geometry": {"filename": "package://" + package_name + "/" + dir + "" + cur_link.name + ".stl"},
                "material": "grey"
            }
            link_collision = {
                "origin": {"xyz": ' '.join(map(str, visual_pos)), "rpy": '0 0 0'},
                "geometry": {"filename": "package://" + package_name + "/" + dir + "" + cur_link.name + ".stl"}
            }

            per_voxel_mass = self.args.voxel_density * (self.args.voxel_size ** 3)
            part_mass = per_voxel_mass * self.mesh_group.get_voxels(cur_link.name).shape[0]
            inetial_matrix, CoM = calculate_inertia_tensor(self.mesh_group.get_voxels(cur_link.name) / 100.0, part_mass, np.eye(4))
            link_inertial = {
                "origin": {"xyz": ' '.join(map(str, CoM + (rel_pos - motor_pos) / 100.0)), "rpy": '0 0 0'},
                "mass": str(self.mesh_group.get_voxels(cur_link.name).shape[0] * self.args.voxel_density),
                "inertia": {"ixx": inetial_matrix[0, 0], "iyy": inetial_matrix[1, 1], "izz": inetial_matrix[2, 2], "ixy": inetial_matrix[0, 1], "ixz": inetial_matrix[0, 2], "iyz": inetial_matrix[1, 2]}
            }
            write_link(urdf_file=urdf_file, link_name=cur_link.name, visual=link_visual, collision=link_collision, inertial=link_inertial)
            
            per_voxel_mass = self.args.voxel_density * (self.args.voxel_size ** 3)
            self.ideal_mass += self.mesh_group.get_voxels(cur_link.name).shape[0] * per_voxel_mass

            if cur_link.name != "BODY":
                # Write a 2DoF joint
                if len(cur_link.axis) == 3:
                    write_link(urdf_file=urdf_file, link_name=cur_link.name + '_virtual')
                    joint1 = {
                        "joint_name": cur_link.name + '_joint1',
                        "joint_type": "revolute",
                        "parent_link": self.father_link_dict[cur_link.name],
                        "child_link": cur_link.name + '_virtual',
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, motor_direct))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **joint1)
                    motor2_pos, motor2_direct, motor_radius = self.link_motor_dict[cur_link.name][0]
                    joint2 = {
                        "joint_name": cur_link.name + '_joint2',
                        "joint_type": "revolute",
                        "parent_link": cur_link.name + '_virtual',
                        "child_link": cur_link.name,
                        "origin": {"xyz": ' '.join(map(str, (motor2_pos - motor_pos) / 100.0)), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, motor2_direct))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **joint2)


                # Write a 1 DoF joint
                elif len(cur_link.axis) == 2:
                    cur_joint = {
                        "joint_name": cur_link.name + '_joint',
                        "joint_type": "revolute",
                        "parent_link": self.father_link_dict[cur_link.name],
                        "child_link": cur_link.name,
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, motor_direct))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **cur_joint)
            
        # Writre foot link
        for contact_node in self.contact_nodes:
            for joint_name, joint_pos in contact_node.val.joints.items():
                if 'foot' in joint_name:
                    write_link(urdf_file=urdf_file, link_name=joint_name)
                    rel_pos = np.array(np.array(joint_pos) - contact_node.val.axis[0]) / 100.0
                    foot_joint = {
                        "joint_name": joint_name + '_joint',
                        "joint_type": "fixed",
                        "parent_link": contact_node.val.name,
                        "child_link": joint_name,
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": "0 0 0"},
                        "limit": {"lower": "0", "upper": "0", "effort": "0", "velocity": "0"}
                    }
                    write_joint(urdf_file=urdf_file, **foot_joint)
                    break

        urdf_file.write('<gazebo>\n')
        urdf_file.write('   <plugin filename="libgazebo_ros_p3d.so" name="p3d_base_controller">\n')
        urdf_file.write('   <alwaysOn>true</alwaysOn>\n')
        urdf_file.write('       <updateRate>10.0</updateRate>\n')
        urdf_file.write('       <bodyName>base_link</bodyName>\n')
        urdf_file.write('       <topicName>odom/ground_truth</topicName>\n')
        urdf_file.write('       <gaussianNoise>0.00000001</gaussianNoise>\n')
        urdf_file.write('       <frameName>world</frameName>\n')
        urdf_file.write('       <xyzOffsets>0 0 0</xyzOffsets>\n')
        urdf_file.write('       <rpyOffsets>0 0 0</rpyOffsets>\n')
        urdf_file.write('   </plugin>\n')
        urdf_file.write('</gazebo>\n')
        urdf_file.write('<gazebo>\n')
        urdf_file.write('   <plugin filename="libgazebo_ros_control.so" name="gazebo_ros_control">\n')
        urdf_file.write('       <legacyModeNS>true</legacyModeNS>\n')
        urdf_file.write('   </plugin>\n')
        urdf_file.write('</gazebo>\n')
        urdf_file.write('</robot>\n')

        # Return the dir of urdf file
        return self.urdf_dir

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=40, help='The expected width of the model')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='The size of the voxel')
    parser.add_argument('--voxel_density', type=float, default=2e-4, help='The density of the voxel')
    args = parser.parse_args()
    save_pre_result = False
    motor_lib = [[3.6, 3.8, 12],  # DM6006         # Height, Radius, Torque
                 [4.5, 2.5, 7 ]]  # DM4310
    if save_pre_result:
        mesh_loader = Custom_Mesh_Loader(args)
        mesh_dir = os.path.normpath('./model/given_models/' + args.model_name + '.stl')
        joint_dir = os.path.normpath('./model/given_models/' + args.model_name + '_joints.pkl')
        mesh_loader.load_mesh(mesh_dir)
        mesh_loader.load_joint_positions(joint_dir)
        mesh_loader.scale()

        mesh_decomp = Mesh_Decomp(args, mesh_loader)
        mesh_decomp.decompose()
        bounds = np.array(get_bounds(mesh_decomp.link_tree, threshold=3)).reshape(-1, 2)
        
        
        motor_opt = Motor_Opt(args, mesh_decomp, None, None)
        motor_opt.motor_results = np.load('./results/' + args.model_name + '_motor_results.npy')
        # motor_opt.render()
        
        joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
        joint_connect_opt.run_opt()
        mesh_decomp.render()
        pkl.dump(mesh_decomp.mesh_group, open('./results/' + args.model_name + '_mesh_group.pkl', 'wb'))
        pkl.dump(mesh_decomp.link_tree, open('./results/' + args.model_name + '_link_tree.pkl', 'wb'))
        pkl.dump(mesh_decomp.father_link_dict, open('./results/' + args.model_name + '_father_link_dict.pkl', 'wb'))
    
    else:
        mesh_group = pkl.load(open('./results/' + args.model_name + '_mesh_group.pkl', 'rb'))
        link_tree = pkl.load(open('./results/' + args.model_name + '_link_tree.pkl', 'rb'))
        father_link_dict = pkl.load(open('./results/' + args.model_name + '_father_link_dict.pkl', 'rb'))
        motor_results = np.load('./results/' + args.model_name + '_motor_results.npy')
        interference_removal = InterferenceRemoval(args=args, 
                                                   mesh_group=mesh_group, 
                                                   motor_param_result=motor_results, 
                                                   link_tree=link_tree, 
                                                   father_link_dict=father_link_dict)
        
        # pkl.dump(robot_result, open('./results/' + args.model_name + '_robot_result.pkl', 'wb'))
        joint_limits = np.vstack([np.array([-1.0, 1.0]) for _ in range(2*len(motor_results))])
        interference_removal.set_joint_limit(joint_limits)
        interference_removal.remove_interference()
        interference_removal.mesh_group.render()
        urdf_dir = interference_removal.generate_urdf()
        robot_result = RobotOptResult(interference_removal, urdf_dir, motor_lib)