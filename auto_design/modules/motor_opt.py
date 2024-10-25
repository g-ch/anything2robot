"""Motor Optimization Module

This module contains the functions to optimize the motor parameters including motor positions and types. 
The optimization problem is solved by genetic algorithm.

Author: Moji Shi
Date: 2024-03-01
"""
import argparse
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import os
import heapq
import matplotlib.pyplot as plt
import pickle as pkl
from mesh_decomp import Mesh_Decomp, is_points_in_cylinder, is_points_in_shell_top, is_points_in_sphere
from mesh_loader import Custom_Mesh_Loader
from generic import Generic_Algorithm, Improved_Generic_Algorithm
from plot_utils import rotation_matrix_from_vectors
from collision_check import check_collision
from sklearn import svm
import math

import multiprocessing

def set_diff_numpy(A, B):
    # Create an array of shape (A.shape[0], B.shape[0]) where each element in A is compared with each in B
    # This results in a boolean array where True indicates that an element of A exists in B
    A_ext = np.expand_dims(A, axis=1)  # Extend A to (n, 1, 3)
    matches = np.any(np.all(A_ext == B, axis=2), axis=1)  # Compare A extended with B and collapse dimensions
    
    # Use the boolean array to filter elements in A that are not in B
    diff = A[~matches]  # Select those elements of A which do not match any element in B
    
    return diff

def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def a_star_search(voxel_3D, start_idx, end_idxs, collision_values):
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    queue = []
    heapq.heappush(queue, (0, start_idx))
    costs = {start_idx: 0}
    parent_dict = {start_idx: None}

    while queue:
        current_cost, cur_idx = heapq.heappop(queue)

        # Check if we have reached the goal
        if np.any(np.all(end_idxs == cur_idx, axis=1)):
            break
        
        # Explore each possible direction
        for direction in directions:
            new_idx = (cur_idx[0] + direction[0], cur_idx[1] + direction[1], cur_idx[2] + direction[2])
            
            # Check if new index is within the voxel grid bounds
            if (0 <= new_idx[0] < voxel_3D.shape[0] and
                0 <= new_idx[1] < voxel_3D.shape[1] and
                0 <= new_idx[2] < voxel_3D.shape[2] and
                voxel_3D[new_idx] not in collision_values):
                
                new_cost = current_cost + 1  # Assuming uniform cost for simplicity
                
                # If new node has not been visited or a cheaper path to it is found
                if new_idx not in costs or new_cost < costs[new_idx]:
                    costs[new_idx] = new_cost
                    priority = new_cost + heuristic(new_idx, np.mean(end_idxs, axis=0))
                    heapq.heappush(queue, (priority, new_idx))
                    parent_dict[new_idx] = cur_idx

    # Reconstruct the path from end to start by following parent links
    path = []
    if np.any(np.all(end_idxs == cur_idx, axis=1)):
        while cur_idx:
            path.append(cur_idx)
            cur_idx = parent_dict[cur_idx]
        path.reverse()

    return path

class General_GA(Improved_Generic_Algorithm):
    def __init__(self, bounds, int_bounds, joint_tree, mesh_decomp, motor_type_params,
                 genome_length, 
                 generation_num, 
                 population_size, 
                 mutation_rate, 
                 crossover_rate,
                 connector_lib) -> None:
        super().__init__(bounds, int_bounds, genome_length, generation_num, population_size, mutation_rate, crossover_rate)
        self.joint_tree = joint_tree
        self.mesh = mesh_decomp.mesh
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh_decomp.mesh.mesh_o3d))
        self.motor_type_params = motor_type_params
        self.connector_lib = connector_lib
        self.father_link_dict = mesh_decomp.father_link_dict
        
        # Get the initial state from bounds
        continuous_mean = np.mean(np.array(bounds), axis=1)
        continuous_var = 1
        self.initial_population = [self.encode(np.concatenate((continuous_mean + continuous_var * np.clip(np.random.randn(len(bounds)), -1, 1), 
                                                               np.array([int_bounds[i][0] for i in range(len(int_bounds))])))) for _ in range(population_size)]

    def get_motor_params(self, genome):
        x = self.decode(genome)
        motor_num = len(x) // 4
        motor_positions = []
        motor_types = []
        motor_directs = []

        queue = [self.joint_tree]
        link_idx = 0
        while queue:
            cur_node = queue.pop(0)
            for child_node in cur_node.children:
                queue.append(child_node)
            if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
                continue

            motor_position = np.array(x[3 * link_idx : 3 * link_idx + 3])
            motor_positions.append(motor_position)
            motor_types.append(x[3 * motor_num + link_idx])
            motor_directs.append(np.array(cur_node.val.axis[1]))

            if len(cur_node.val.axis) == 3:
                motor_idx = int(x[3 * motor_num + link_idx])
                motor2_position = motor_position - self.connector_lib[motor_idx][0] * np.array(cur_node.val.axis[1]) + self.connector_lib[motor_idx][1] * np.array(cur_node.val.axis[2])
                motor_positions.append(motor2_position)
                motor_types.append(x[3 * motor_num + link_idx])
                motor_directs.append(np.array(cur_node.val.axis[2]))
            
            link_idx += 1

        return np.array(motor_positions), np.array(motor_directs), np.array(motor_types, dtype=int)
    
    def get_occupancy_cost(self, motor_poses, motor_directs, motor_types):
        """
        Calculate the occupancy cost of the motors according to the SDF of the mesh
        """
        
        def sample_points_in_cylinder(center, axis_dir, r, h):
            axis_dir = axis_dir / np.linalg.norm(axis_dir)
            if (axis_dir == np.array([0, 0, 1])).all():
                ortho1 = np.array([1, 0, 0])
            else:
                ortho1 = np.cross(axis_dir, [0, 0, 1])
                ortho1 /= np.linalg.norm(ortho1)
            ortho2 = np.cross(axis_dir, ortho1)


            rho_values = np.arange(r, r+0.001, r)
            theta_values = np.arange(0, 2 * np.pi, np.pi / 4)
            z_values = np.arange(0, h+0.001, h)

            rho, theta, z = np.meshgrid(rho_values, theta_values, z_values, indexing='ij')
            x = rho * np.cos(theta)
            y = rho * np.sin(theta)
            points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
            rotation_matrix = np.column_stack((ortho1, ortho2, axis_dir))
            transformed_points = points @ rotation_matrix.T + center
            unique_points = np.unique(transformed_points, axis=0)
            return unique_points

        # cost = 0

        # # Conduct BFS for cost
        # queue = [self.joint_tree]
        # cur_idx = 0
        # while queue:
        #     cur_node = queue.pop(0)
        #     for child_node in cur_node.children:
        #         queue.append(child_node)
        #     if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
        #         continue
        #     motor_position = motor_poses[cur_idx]
        #     motor_type = int(motor_types[cur_idx])
        #     motor_direct = np.array(cur_node.val.axis[1])

        #     target_mesh = o3d.io.read_triangle_mesh("./urdf/lynel/tmp/" + self.father_link_dict[cur_node.val.name].name + '_ideal.stl')
        #     target_mesh.scale(100, center=(0, 0, 0))
        #     cur_scene = o3d.t.geometry.RaycastingScene()
        #     _ = cur_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target_mesh))

        #     points = sample_points_in_cylinder(motor_position, motor_direct, self.motor_type_params[motor_type][1], self.motor_type_params[motor_type][0])
        #     cur_idx += 1

        #     # If the joint has 2 axis, then it has 2 motors connected by a fixed-sized connector
        #     if len(cur_node.val.axis) == 3:
        #         motor2_pos = motor_position - self.connector_params[0] * np.array(cur_node.val.axis[1]) + self.connector_params[1] * np.array(cur_node.val.axis[2])
        #         motor2_direct = np.array(cur_node.val.axis[2])
        #         motor2_type = motor_type
        #         points = np.vstack((points, sample_points_in_cylinder(motor2_pos, motor2_direct, self.motor_type_params[motor2_type][1], self.motor_type_params[motor2_type][0])))
        #         cur_idx += 1
            
        #     points_tensor = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
        #     all_distances = cur_scene.compute_signed_distance(points_tensor).numpy()
            
        #     # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        #     # pcd = o3d.geometry.PointCloud()
        #     # pcd.points = o3d.utility.Vector3dVector(points)
        #     # o3d.visualization.draw_geometries([target_mesh, axis, pcd])
            
        #     cost += np.mean(all_distances)



        all_points = np.empty((0, 3), dtype=np.float32)
        for i in range(len(motor_poses)):
            points = sample_points_in_cylinder(motor_poses[i], motor_directs[i], self.motor_type_params[motor_types[i]][1], self.motor_type_params[motor_types[i]][0])
            all_points = np.vstack((all_points, points))
        points_tensor = o3d.core.Tensor(all_points, dtype=o3d.core.Dtype.Float32)
        all_distances = self.scene.compute_signed_distance(points_tensor).numpy()
        all_distances = all_distances.reshape(-1, points.shape[0])
        cost = np.mean(np.max(all_distances, axis=1))
        

        # cost = 0
        # for i in range(len(motor_poses)):
        #     points_tensor = o3d.core.Tensor(np.array(motor_poses[i]).reshape(-1, 3), dtype=o3d.core.Dtype.Float32)
        #     all_distances = self.scene.compute_signed_distance(points_tensor).numpy()
        #     cost += np.max(all_distances)

        return cost

    def get_position_cost(self, distance, sigmoidal=False):
        if sigmoidal:
            return (np.exp(distance) - 1)
        return distance

    def check_constraint(self, motor_positions, motor_directs, motor_types):
        # 1. Check the collision between motors
        for i in range(len(motor_positions)):
            for j in range(i+1, len(motor_positions)):
                if i != j:
                    cylinder1 = {'center': motor_positions[i], 
                                'direct': motor_directs[i], 
                                'height': self.motor_type_params[motor_types[i]][0], 
                                'radius': self.motor_type_params[motor_types[i]][1]}
                    cylinder2 = {'center': motor_positions[j], 
                                'direct': motor_directs[j],
                                'height': self.motor_type_params[motor_types[j]][0],
                                'radius': self.motor_type_params[motor_types[j]][1]}
                    flag_collision, _ = check_collision(cylinder1, cylinder2)
                    if flag_collision:
                        return True
        
        # 2. Check the motor torque
        return False
    

    def get_costs(self, genome):
        motor_positions, motor_directs, motor_types = self.get_motor_params(genome)
        if self.check_constraint(motor_positions, motor_directs, motor_types):
            return 500000, 500000
        
        cost_motor_position = 0
        cost_motor_occupancy = 0

        # Conduct BFS for cost
        queue = [self.joint_tree]
        cur_idx = 0
        while queue:
            cur_node = queue.pop(0)
            for child_node in cur_node.children:
                queue.append(child_node)
            if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
                continue
            motor_position = motor_positions[cur_idx]
            motor_type = int(motor_types[cur_idx])
            motor_direct = np.array(cur_node.val.axis[1])

            cur_idx += 1

            # Positional Cost
            ## Linear Positional Cost
            cost_motor_position += 10 * np.linalg.norm(motor_position - np.array(cur_node.val.axis[0]))
            ## Sigmoidal Positional Cost
            # cost_motor_position += 10 * self.get_position_cost(np.linalg.norm(motor_position - np.array(cur_node.val.axis[0])), sigmoidal=False)


            # If the joint has 2 axis, then it has 2 motors connected by a fixed-sized connector
            if len(cur_node.val.axis) == 3:
                motor_idx = motor_type
                motor2_pos = motor_position - self.connector_lib[motor_idx][0] * np.array(cur_node.val.axis[1]) + self.connector_lib[motor_idx][1] * np.array(cur_node.val.axis[2])
                motor2_direct = np.array(cur_node.val.axis[2])
                motor2_type = motor_type
                # cost_motor_position += 0.5 * self.get_position_cost(np.linalg.norm(motor2_pos - np.array(cur_node.val.axis[0])), sigmoidal=False)
                cur_idx += 1
        
        # Occupancy Cost
        cost_motor_occupancy += 10 * self.get_occupancy_cost(motor_positions, 
                                             motor_directs, 
                                             motor_types)
        
        return cost_motor_position, cost_motor_occupancy



    def fitness_function(self, genome) -> float:
        
        cost_motor_position, cost_motor_occupancy = self.get_costs(genome) 
        cost = cost_motor_position + cost_motor_occupancy

        return cost

    def from_genome_to_motor_results(self, genome):
        results = []
        motor_positions, motor_directs, motor_types = self.get_motor_params(genome)

        for i in range(len(motor_positions)):
            base = motor_positions[i] - motor_directs[i] * self.motor_type_params[motor_types[i]][0] / 2
            top = motor_positions[i] + motor_directs[i] * self.motor_type_params[motor_types[i]][0] / 2
            results.extend([*base, *top, self.motor_type_params[motor_types[i]][1]])

        # queue = [self.joint_tree]
        # cur_idx = 0

        # while queue:
        #     cur_node = queue.pop(0)
        #     for child_node in cur_node.children:
        #         queue.append(child_node)
        #     if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
        #         continue
            
        #     motor_position = motor_positions[cur_idx]
        #     motor_direct = np.array(cur_node.val.axis[1])
        #     motor_type = int(motor_types[cur_idx])


        #     base = motor_position - motor_direct * self.motor_type_params[motor_type][0] / 2
        #     top = motor_position + motor_direct * self.motor_type_params[motor_type][0] / 2
        #     results.extend([*base, *top, self.motor_type_params[motor_type][1]])

        #     if len(cur_node.val.axis) == 3:
        #         motor2_pos = motor_position - self.connector_params[0] * np.array(cur_node.val.axis[1]) + self.connector_params[1] * np.array(cur_node.val.axis[2])
        #         motor2_direct = np.array(cur_node.val.axis[2])
        #         motor2_type = motor_type

        #         base = motor2_pos - motor2_direct * self.motor_type_params[motor2_type][0] / 2
        #         top = motor2_pos + motor2_direct * self.motor_type_params[motor2_type][0] / 2
        #         results.extend([*base, *top, self.motor_type_params[motor2_type][1]])
            
        #     cur_idx += 1

        return np.array(results).reshape(-1, 7)
        

class Motor_Opt:
    def __init__(self, args, mesh_decomp : Mesh_Decomp, bounds, motor_lib, connector_lib):
        self.args = args
        self.ga_runner = None
        self.mesh_decomp = mesh_decomp
        self.mesh = mesh_decomp.mesh
        self.bounds = bounds
        self.motor_lib = motor_lib
        self.connector_lib = connector_lib
    
    def choose_motor_type(self):
        joint_names, max_torques = self.mesh_decomp.generate_constraints()
        torque_dict = {}
        for i in range(len(joint_names)):
            print("Joint Name: ", joint_names[i], "Max Torque: ", max_torques[i])
            torque_dict[joint_names[i]] = max_torques[i]

        motor_types = []
        motor_lib = np.array(self.motor_lib)
        queue = [self.mesh_decomp.link_tree]
        while queue:
            cur_node = queue.pop(0)
            for child_node in cur_node.children:
                queue.append(child_node)
            if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
                continue

            # From the motor list, choose the motor type that satisfies the constraints
            if len(cur_node.val.axis) == 2:
                suitable_motors = np.where(motor_lib[:, 2] - torque_dict[cur_node.val.name+'_joint'] > 0)[0]
                if len(suitable_motors) == 0:
                    raise ValueError(f"No motor in the library meets the torque requirement for {cur_node.val.name+'_joint'}. Required torque: {torque_dict[cur_node.val.name+'_joint']}, Max available torque: {motor_lib[:, 2].max()}")
                motor_type = np.where(motor_lib[:, 2] == motor_lib[suitable_motors, 2].min())[0][0]
                motor_types.append(motor_type)

            # If the joint has 2 axis, choose the motor type that satisfies the torque limit of both motors
            if len(cur_node.val.axis) == 3:
                # Find motors that meet the torque requirement
                suitable_motors = np.where(motor_lib[:, 2] - torque_dict[cur_node.val.name+'_joint1'] > 0)[0]
                if len(suitable_motors) == 0:
                    raise ValueError(f"No motor in the library meets the torque requirement for {cur_node.val.name+'_joint1'}. Required torque: {torque_dict[cur_node.val.name+'_joint1']}, Max available torque: {motor_lib[:, 2].max()}")
                # Select the motor with the lowest sufficient torque
                motor_type1 = np.where(motor_lib[:, 2] == motor_lib[suitable_motors, 2].min())[0][0]

                suitable_motors = np.where(motor_lib[:, 2] - torque_dict[cur_node.val.name+'_joint2'] > 0)[0]
                if len(suitable_motors) == 0:
                    raise ValueError(f"No motor in the library meets the torque requirement for {cur_node.val.name+'_joint2'}. Required torque: {torque_dict[cur_node.val.name+'_joint2']}, Max available torque: {motor_lib[:, 2].max()}")
                motor_type2 = np.where(motor_lib[:, 2] == motor_lib[suitable_motors, 2].min())[0][0]

                if motor_lib[motor_type1][2] > motor_lib[motor_type2][2]:
                    motor_types.append(motor_type1)
                else:
                    motor_types.append(motor_type2)

        return np.array(motor_types)

    def run_opt(self, generation_num=10):
        self.motor_types = self.choose_motor_type()
        self.ga_runner = General_GA(bounds=self.bounds, 
                                    int_bounds=[[self.motor_types[i]] for i in range(self.motor_types.shape[0])],
                                    joint_tree=self.mesh_decomp.link_tree, 
                                    mesh_decomp=self.mesh_decomp, 
                                    motor_type_params=self.motor_lib,
                                    genome_length=500, 
                                    generation_num=generation_num, 
                                    population_size=100, 
                                    mutation_rate=0.05, 
                                    crossover_rate=0.3,
                                    connector_lib=self.connector_lib)

        # Generate initial state where motors are put right at the position of relevant joints
        genome_result, cost_log, best_fitness = self.ga_runner.run_generic(self.ga_runner.initial_population)
        self.motor_results = self.ga_runner.from_genome_to_motor_results(genome_result)
        return self.motor_results, cost_log, best_fitness

    def create_motors(self, 
                      motor_params_results,
                      colors = ['#2DB3F0', '#8E75AF', '#C03027', '#748d71']):
        objs = []
        
        for i in range(motor_params_results.shape[0]):
            c1 = motor_params_results[i][:3]
            c2 = motor_params_results[i][3:6]
            r = motor_params_results[i][6]

            # Calculate direction vector and height
            direction = c2 - c1
            h = np.linalg.norm(direction)
            direction /= h  # Normalize direction vector

            # Rotation matrix to align circle normal to the cylinder direction
            rot_matrix = rotation_matrix_from_vectors(np.array([0, 0, 1]), direction)

            # Generate cylinder surface
            theta = np.linspace(0, 2 * np.pi, 100)
            steps = 10  # Number of steps along the cylinder's height
            for step in np.linspace(0, 1, steps):
                circle_x = r * np.cos(theta)
                circle_y = r * np.sin(theta)
                circle_z = np.zeros_like(theta)  # Initially, circles are in the xy-plane
                circle_points = np.vstack((circle_x, circle_y, circle_z)).T
                circle_points = circle_points @ rot_matrix.T  # Apply rotation
                circle_points += c1 + direction * step * h  # Translate to position

                cylinder_surface = go.Scatter3d(x=circle_points[:, 0], y=circle_points[:, 1], z=circle_points[:, 2],
                                                mode='lines', line=dict(color="red", width=3),
                                                showlegend=False)
                objs.append(cylinder_surface)
        return objs
    

    def save_fig(self, fig, save_path):
        """Save the figure to a file."""
        try:
            fig.write_image(save_path)
        except Exception as e:
            print(f"Error while saving the figure: {e}")

    def render(self, save_only=False, save_path=None):
        fig = go.Figure(data=[self.mesh.mesh_plotly, *self.create_motors(self.motor_results)])
        fig.update_layout(scene_aspectmode='data')

        if not save_only:
            fig.show()
        if save_path is not None:
            # Create a separate process for saving the figure
            process = multiprocessing.Process(target=self.save_fig, args=(fig, save_path))
            process.start()
            
            # Wait for the process to complete or timeout
            timeout = 30  # Timeout in seconds
            process.join(timeout)
            
            if process.is_alive():
                print("Saving the figure took too long! Terminating the process...")
                process.terminate()  # Forcefully kill the process
                process.join()       # Ensure the process is terminated
            else:
                if os.path.exists(save_path):
                    print(f"Image saved at: {save_path}")
                else:
                    print("Saving failed.")

class Joint_Connect_Opt:
    def __init__(self, args, mesh_decomp : Mesh_Decomp, motor_params_results: np.ndarray):
        self.args = args
        self.mesh_decomp = mesh_decomp
        self.mesh = mesh_decomp.mesh
        self.motor_params_results = motor_params_results
        self.motor_shell = 1.5

        self.father_dict = self.mesh_decomp.father_link_dict
        for link_name in self.father_dict:
            self.father_dict[link_name] = self.father_dict[link_name].name

    def connect_voxels(self, mesh_group, start_idxs, end_idxs, connect_link_name):
        path = []
        for i in range(start_idxs.shape[0]):
            added_path = a_star_search(mesh_group.voxel_data, (start_idxs[i,0], start_idxs[i,1], start_idxs[i,2]), end_idxs, collision_values=[0])
            if added_path != []:
                path += added_path
        path = np.array(path).reshape(-1, 3)
        mesh_group.set_voxels(connect_link_name, path, index=True)
        return mesh_group.index_to_position(list(path))

    def run_opt(self):
        queue = [self.mesh_decomp.link_tree]
        cur_idx = 0

        count = 0
        while queue:
            count += 1
            cur_node = queue.pop(0)
            for child_node in cur_node.children:
                queue.append(child_node)
            if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
                continue

            cur_link_name = cur_node.val.name
            print("Joint Connection Opt: ", count, "Current Link Name: ", cur_link_name)
            get_removed_list = lambda list, remove_value: [value for value in list if value != remove_value]
            
            def condition_classification(pts):
                #return is_points_in_sphere(pts, (motor_param[:3] + motor_param[3:6]) / 2, radius=10)
                top_point = motor_param[:3]
                bottom_point = motor_param[3:6]
                top_bottom_dist_half = np.linalg.norm(top_point - bottom_point) / 2
                sphere_radius = math.sqrt(top_bottom_dist_half**2 + motor_param[6]**2)
                return is_points_in_sphere(pts, (motor_param[:3] + motor_param[3:6]) / 2, sphere_radius * 2)

            # Get the motor parameters. The motor parameters are stored in the form of [base (3D position), top (3D position), radius]
            motor_param = self.motor_params_results[cur_idx]
            
            # Given a sphere, the center is the motor's center, find all the voxels that are in the sphere and their types
            classify_voxels = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=[cur_link_name, self.father_dict[cur_link_name]],
                                                                      target_group_name=None,
                                                                      condition_func=condition_classification)
            classify_voxels_values = self.mesh_decomp.mesh_group.get_voxel_type(classify_voxels)

            # set binary values for the voxels
            classify_voxels_values = np.where(classify_voxels_values == self.mesh_decomp.mesh_group.link_value_dict[cur_link_name], 1, 0)

            # Find a planar coordinate that is perpendicular to the motor's direction, the coordinate is defined by x and y axis
            motor_direct = (motor_param[3:6] - motor_param[:3]) / np.linalg.norm(motor_param[3:6] - motor_param[:3])

            x_direct = np.array([1, 0, 0])
            if np.abs(np.dot(motor_direct, x_direct)) > 0.9:
                x_direct = np.array([0, 1, 0])
            x_direct = np.cross(motor_direct, x_direct)
            y_direct = np.cross(motor_direct, x_direct)

            # Project the voxels to the planar coordinate
            projected_voxels = np.dot(classify_voxels - motor_param[:3], np.array([x_direct, y_direct, motor_direct]))[:, :2]

            # SVM to classify the voxels
            clf = svm.LinearSVC(C=1.0, fit_intercept=False, max_iter=100, tol=10, dual=True)
            clf.fit(projected_voxels, classify_voxels_values)

            def svm_predict_with_margin(clf, pts, margin=0.5):
                distances = clf.decision_function(pts)
                svm_result = np.where(
                    np.abs(distances) < margin, 
                    -1,  # -1 for points near the decision boundary
                    clf.predict(pts)  # Original class labels for others
                )
                return svm_result
            
            def condition_child_link_radical(pts):
                projected_pts = np.dot(pts - motor_param[:3], np.array([x_direct, y_direct, motor_direct]))[:, :2]
                #svm_result = clf.predict(projected_pts)
                margin = motor_param[6] * 0.5
                svm_result = svm_predict_with_margin(clf, projected_pts, margin)
                return np.logical_and(is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 0.0, self.motor_shell), 
                                      svm_result == 1)
            def condition_father_link_radical(pts):
                projected_pts = np.dot(pts - motor_param[:3], np.array([x_direct, y_direct, motor_direct]))[:, :2]
                # svm_result = clf.predict(projected_pts)
                margin = motor_param[6] * 0.5
                svm_result = svm_predict_with_margin(clf, projected_pts, margin)
                return np.logical_and(is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 0.0, self.motor_shell), 
                                      svm_result == 0)

            def condition_child_link_top(pts):
                return np.logical_and(is_points_in_shell_top(pts, motor_param[:3], motor_param[3:6], motor_param[6], 1.0, self.motor_shell), 
                                      is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 1.0, self.motor_shell))
            def condition_father_link_top(pts):
                return np.logical_and(is_points_in_shell_top(pts, motor_param[3:6], motor_param[:3], motor_param[6], 1.0, self.motor_shell), 
                                      is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 1.0, self.motor_shell))

        
            if len(cur_node.val.axis) == 2:  # One DOF axis

                # Add voxels to father link     
                father_link_addition_voxels_top = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=list(self.mesh_decomp.mesh_group.link_value_dict.keys()),
                                                                                        target_group_name=self.father_dict[cur_link_name],
                                                                                        condition_func=condition_father_link_top)
                father_link_addition_voxels_radical = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_decomp.mesh_group.link_value_dict.keys()), self.father_dict[cur_link_name]),
                                                                                            target_group_name=self.father_dict[cur_link_name],
                                                                                            condition_func=condition_father_link_radical)
                
                # Add voxels to child link
                child_link_addition_voxels_top = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=list(self.mesh_decomp.mesh_group.link_value_dict.keys()),
                                                                                        target_group_name=cur_link_name,
                                                                                        condition_func=condition_child_link_top)
                child_link_addition_voxels_radical = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_decomp.mesh_group.link_value_dict.keys()), cur_link_name),
                                                                                            target_group_name=cur_link_name,
                                                                                            condition_func=condition_child_link_radical)
                
                
                non_removal_voxels = np.vstack((father_link_addition_voxels_top, child_link_addition_voxels_top))
                non_removal_voxels = np.unique(non_removal_voxels, axis=0)
                non_removal_indices = self.mesh_decomp.mesh_group.position_to_index(non_removal_voxels)
                self.mesh_decomp.mesh_group.voxel_no_removal[non_removal_indices[:,0], non_removal_indices[:,1], non_removal_indices[:,2]] = 1

                # Connect the addictive child link voxels to child link
                start_idx = self.mesh_decomp.mesh_group.position_to_index(child_link_addition_voxels_top)
                target_positions = set_diff_numpy(self.mesh_decomp.mesh_group.get_voxels(cur_link_name), np.vstack((child_link_addition_voxels_radical, child_link_addition_voxels_top)))
                end_idxs = self.mesh_decomp.mesh_group.position_to_index(target_positions)
                self.connect_voxels(self.mesh_decomp.mesh_group, start_idx, end_idxs, cur_link_name)

                # Connect the addictive father link voxels to father link
                father_link_top_voxels = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=list(self.mesh_decomp.mesh_group.link_value_dict.keys()),
                                                                                target_group_name=None,
                                                                                condition_func=condition_father_link_top)
                start_idx = self.mesh_decomp.mesh_group.position_to_index(father_link_top_voxels)
                target_positions = set_diff_numpy(self.mesh_decomp.mesh_group.get_voxels(self.father_dict[cur_link_name]), np.vstack((father_link_addition_voxels_radical, father_link_addition_voxels_top)))
                end_idxs = self.mesh_decomp.mesh_group.position_to_index(target_positions)
                self.connect_voxels(self.mesh_decomp.mesh_group, start_idx, end_idxs, self.father_dict[cur_link_name])
                cur_idx += 1

            elif len(cur_node.val.axis) == 3: # Two DOF axis

                # def condition_shell(pts):
                #     return is_points_in_cylinder(pts, motor_param[:3], motor_param[3:6], motor_param[6], 1.0, 0)
                # check_voxels = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_decomp.mesh_group.link_value_dict.keys()), self.father_dict[cur_link_name]),
                #                                                        target_group_name=cur_link_name,
                #                                                        condition_func=condition_shell)
                # Connect the addictive child link voxels to child link
                child_link_addition_voxels_top = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=list(self.mesh_decomp.mesh_group.link_value_dict.keys()),
                                                                                         target_group_name=cur_link_name,
                                                                                         condition_func=condition_child_link_top)
                
                child_link_addition_voxels_radical = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_decomp.mesh_group.link_value_dict.keys()), cur_link_name),
                                                                                            target_group_name=cur_link_name,
                                                                                            condition_func=condition_child_link_radical)
                
                start_idx = self.mesh_decomp.mesh_group.position_to_index(child_link_addition_voxels_top)
                # target_positions = set_diff_numpy(self.mesh_decomp.mesh_group.get_voxels(cur_link_name), child_link_addition_voxels_top)

                target_positions = set_diff_numpy(self.mesh_decomp.mesh_group.get_voxels(cur_link_name), np.vstack((child_link_addition_voxels_radical, child_link_addition_voxels_top)))

                end_idxs = self.mesh_decomp.mesh_group.position_to_index(target_positions)
                added_voxels = self.connect_voxels(self.mesh_decomp.mesh_group, start_idx, end_idxs, cur_link_name)

                
                # Connect the addictive father link voxels to father link
                motor_param = self.motor_params_results[cur_idx + 1]
                # check_voxels = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=get_removed_list(list(self.mesh_decomp.mesh_group.link_value_dict.keys()), self.father_dict[cur_link_name]),
                #                                         target_group_name=self.father_dict[cur_link_name],
                #                                         condition_func=condition_shell)
                father_link_addition_voxels_top = self.mesh_decomp.mesh_group.move_voxels(initial_group_names=list(self.mesh_decomp.mesh_group.link_value_dict.keys()),
                                                                                          target_group_name=self.father_dict[cur_link_name],
                                                                                          condition_func=condition_child_link_top)
                non_removal_voxels = np.vstack((father_link_addition_voxels_top, child_link_addition_voxels_top, added_voxels))
                non_removal_voxels = np.unique(non_removal_voxels, axis=0)
                non_removal_indices = self.mesh_decomp.mesh_group.position_to_index(non_removal_voxels)
                self.mesh_decomp.mesh_group.voxel_no_removal[non_removal_indices[:,0], non_removal_indices[:,1], non_removal_indices[:,2]] = 1

                
                start_idx = self.mesh_decomp.mesh_group.position_to_index(father_link_addition_voxels_top)
                target_positions = set_diff_numpy(self.mesh_decomp.mesh_group.get_voxels(self.father_dict[cur_link_name]), father_link_addition_voxels_top)
                end_idxs = self.mesh_decomp.mesh_group.position_to_index(target_positions)
                cur_idx += 2
            
        
        # render all paths

        # fig = go.Figure(data=[self.mesh.mesh_plotly])
        # for path in all_path:
        #     fig.add_trace(go.Scatter3d
        #         (x=path[:, 0], y=path[:, 1], z=path[:, 2], mode='lines', line=dict(color='red', width=5)))
        # fig.show()

def get_bounds(link_tree, threshold=5):
    """
    Get bounds for motor optimization
    """
    queue = [link_tree]
    bounds = []
    cur_idx = 0

    while queue:
        cur_node = queue.pop(0)
        for child_node in cur_node.children:
            queue.append(child_node)
        if cur_node.val.axis is None or np.linalg.norm(cur_node.val.axis[1]) == 0:
            continue

        axis_pos = np.array(cur_node.val.axis[0])
        bounds.append([axis_pos[0] - threshold,
                       axis_pos[0] + threshold,
                       axis_pos[1] - threshold,
                       axis_pos[1] + threshold,
                       axis_pos[2] - threshold,
                       axis_pos[2] + threshold])

        cur_idx += 1

    return np.array(bounds).reshape(-1, 2)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=40, help='The expected width of the model')
    parser.add_argument('--voxel_size', type=float, default=1.0, help='The size of the voxel')
    parser.add_argument('--voxel_density', type=float, default=1e-4, help='The density of the voxel')
    args = parser.parse_args()
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '.stl')
    joint_dir = os.path.normpath('./auto_design/model/given_models/' + args.model_name + '_joints.pkl')
    mesh_loader.load_mesh(mesh_dir)
    mesh_loader.load_joint_positions(joint_dir)

    mesh_loader.scale()
    # mesh_loader.render()

    mesh_decomp = Mesh_Decomp(args, mesh_loader)
    mesh_decomp.decompose()
    # mesh_decomp.render()
    bounds = np.array(get_bounds(mesh_decomp.link_tree, threshold=6)).reshape(-1, 2)
    motor_lib = [[3.6, 3.8, 12],  # DM6006         # Height, Radius, Torque
                #  [4.5, 2.5, 8 ],  # DM4310
                 [3.75, 4.8, 20 ]]  # DM4310
    motor_opt = Motor_Opt(args, mesh_decomp, bounds, motor_lib)
    motor_results, __, __ = motor_opt.run_opt()
    # np.save('./results/' + args.model_name + '_motor_results1.npy', motor_results)
    motor_opt.render()
    
    # motor_opt = Motor_Opt(args, mesh_decomp, None, None)
    # motor_opt.motor_results = np.load('./results/' + args.model_name + '_motor_results1.npy')
    # motor_opt.render()
    
    joint_connect_opt = Joint_Connect_Opt(args, mesh_decomp, motor_opt.motor_results)
    joint_connect_opt.run_opt()
    mesh_decomp.render()


    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def render_3d_binary_array(data):
        """
        Render a 3D binary numpy array using matplotlib where '1' values are occupied.

        Parameters:
        - data: numpy.ndarray, a 3D binary array where 1 represents an occupied voxel.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Extract the indices of all occupied voxels
        x, y, z = np.where(data == 1)

        # Plot each voxel as a point; you can also use plot_trisurf for surface visualization
        ax.scatter(x, y, z, c='blue', marker='o', s=100)  # s is the size of the point

        # Setting plot limits to match the data array size
        ax.set_xlim([0, data.shape[0]])
        ax.set_ylim([0, data.shape[1]])
        ax.set_zlim([0, data.shape[2]])

        # Labels and title
        ax.set_xlabel('X Dimension')
        ax.set_ylabel('Y Dimension')
        ax.set_zlabel('Z Dimension')
        ax.set_title('3D Visualization of Binary Array')

        plt.show()
    render_3d_binary_array(mesh_decomp.mesh_group.voxel_no_removal)
    # pkl.dump(mesh_decomp.mesh_group, open('./results/' + args.model_name + '_mesh_group.pkl', 'wb'))
    # pkl.dump(mesh_decomp.link_tree, open('./results/' + args.model_name + '_link_tree.pkl', 'wb'))
    # pkl.dump(mesh_decomp.father_link_dict, open('./results/' + args.model_name + '_father_link_dict.pkl', 'wb'))

    