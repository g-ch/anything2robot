"""Mesh Decomposition Module

This module contains the functions to decompose the mesh into different parts.

Author: Moji Shi
Date: 2024-03-01

"""


import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import plotly.graph_objects as go
import argparse
import os
import time
import pinocchio as pin

from mesh_loader import *
from urdf_generator import *
from data_struct import TreeNode, Graph
from scipy.ndimage import binary_dilation
from os.path import abspath, dirname, join
from scipy.ndimage import label, find_objects
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import tqdm


def erode_zeros(arr, structure=None):
    """
    Erodes the zeros in a binary (i,j,k) numpy array, expanding the areas where the value is 0.
    
    Parameters:
        arr (numpy.ndarray): The input binary array with shape (i, j, k).
        structure (numpy.ndarray, optional): The structuring element used for dilation. 
            If None, a simple connectivity structure is used.
    
    Returns:
        numpy.ndarray: The array after erosion of zeros.
    """
    # Invert the array: 0s become 1s, and 1s become 0s
    inverted_arr = np.logical_not(arr)
    
    # Perform dilation on the inverted array
    if structure is None:
        # Default structuring element: connectivity structure (3x3x3)
        dilated_inverted_arr = binary_dilation(inverted_arr)
    else:
        # Use the user-provided structuring element
        dilated_inverted_arr = binary_dilation(inverted_arr, structure=structure)
    
    # Invert back to original form where the original 0s are dilated
    result = np.logical_not(dilated_inverted_arr)
    
    return result

def is_points_in_cylinder(pts, top_center, bottom_center, radius, threshold=2, radius_threshold=2):
    """
    Check if the points are inside the cylinder defined by the top and bottom centers and the radius.
    """
    pts = np.array(pts)
    top_center = np.array(top_center)
    bottom_center = np.array(bottom_center)
    axis_vec = top_center - bottom_center
    pt_vec = pts - bottom_center
    axis_norm = np.linalg.norm(axis_vec)
    proj_lengths = np.dot(pt_vec, axis_vec) / axis_norm
    in_length = (proj_lengths >= -threshold) & (proj_lengths <= axis_norm + threshold)
    distances_to_axis = np.linalg.norm(pt_vec - np.outer(proj_lengths, axis_vec) / axis_norm, axis=1)
    in_radius = distances_to_axis <= radius + radius_threshold

    return in_length & in_radius

def is_points_in_sphere(pts, center, radius, threshold=2):
    """
    Check if the points are inside the sphere defined by the center and the radius.
    """
    pts = np.array(pts)
    center = np.array(center)
    distances = np.linalg.norm(pts - center, axis=1)
    in_sphere = distances <= radius + threshold

    return in_sphere

def is_points_in_shell_top(pts, top_center, bottom_center, radius, threshold=2, radius_threshold=2):
    """
    Check if the points are inside the top shell defined by the top and bottom centers and the radius.
    """
    pts = np.array(pts)
    top_center = np.array(top_center)
    bottom_center = np.array(bottom_center)
    axis_vec = top_center - bottom_center
    pt_vec = pts - bottom_center
    axis_norm = np.linalg.norm(axis_vec)
    proj_lengths = np.dot(pt_vec, axis_vec) / axis_norm
    in_length = (proj_lengths >= -threshold) & (proj_lengths <= 0)
    distances_to_axis = np.linalg.norm(pt_vec - np.outer(proj_lengths, axis_vec) / axis_norm, axis=1)
    in_radius = distances_to_axis <= radius + radius_threshold

    return in_length & in_radius

class Mesh_Group:
    def __init__(self, args) -> None:
        self.voxel_data = None
        self.voxel_no_removal = None
        self.voxel_size = args.voxel_size
        self.color_list = ["#2DB3F0", "#124860", "#8E75AF", "#2b2335", "#C03027", "#4d1310", "#a6caa1", "#748d71", 
                           "grey", "black", "blue", "green", "red", "yellow", "orange", "purple", "pink", 
                           "brown", "cyan", "magenta", "olive", "teal", "navy", "maroon", "lime", "aqua", "fuchsia", 
                           "silver", "gray"]
        self.link_value_dict = {"Unoccupied": 0}
        self.cur_link_value = 1
    
    def set_range(self, x_range, y_range, z_range):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.voxel_data = np.zeros((len(x_range), len(y_range), len(z_range)), dtype=int)
        self.voxel_no_removal = np.zeros((len(x_range), len(y_range), len(z_range)), dtype=int)

    def position_to_index(self, positions):
        """
        Convert the positions to the indices.
        """
        indices = np.zeros_like(positions, dtype=int)
        ranges = [self.x_range, self.y_range, self.z_range]
        
        for i in range(3):  # x, y, z
            start = ranges[i][0]  # Start of the range
            step = self.voxel_size
            indices[:, i] = np.round((positions[:, i] - start) / step).astype(int)  # Convert position to index

        return indices

    def index_to_position(self, indices):
        """
        Convert the indices to the positions.
        """
        positions = np.zeros((len(indices), 3))
        if len(indices) == 0:
            return positions
        indices = np.array(indices)
        for i in range(3):
            positions[:, i] = [self.x_range, self.y_range, self.z_range][i][indices[:, i]]
        return positions
    
    def set_voxels(self, link_name, voxels, index=False):
        if index:
            indices = voxels.astype(int)
        else:
            indices = self.position_to_index(voxels)
        if link_name in self.link_value_dict.keys():
            self.voxel_data[indices[:, 0], indices[:, 1], indices[:, 2]] = self.link_value_dict[link_name]
        else:
            self.voxel_data[indices[:, 0], indices[:, 1], indices[:, 2]] = self.cur_link_value
            self.link_value_dict[link_name] = self.cur_link_value
            self.cur_link_value += 1

    def get_voxels(self, link_name, get_index=False):
        """
        Get the voxels
        """
        indices = np.argwhere(self.voxel_data == self.link_value_dict[link_name])
        if get_index:
            return indices
        else:
            return self.index_to_position(indices)

    def move_voxels(self, initial_group_names, target_group_name, condition_func):
        """
        Move the voxels from the initial group to the target group according to the condition function.
        """

        added_voxels = []
        
        for initial_group_name in initial_group_names:
            conditions = condition_func(self.get_voxels(initial_group_name))
            added_voxels.append(self.get_voxels(initial_group_name)[conditions])
 
        added_voxels = np.vstack(added_voxels)

        if target_group_name is not None:
            self.voxel_data[self.position_to_index(added_voxels)[:, 0], self.position_to_index(added_voxels)[:, 1], self.position_to_index(added_voxels)[:, 2]] = self.link_value_dict[target_group_name]
        
        return added_voxels
    
    def get_voxel_type(self, positions):
        """
        Get the type of the voxel.
        """
        indices = self.position_to_index(positions)
        return self.voxel_data[indices[:, 0], indices[:, 1], indices[:, 2]]

    def render(self, mesh_plotly=None):
        """
        Render the decomposed voxels.
        """
        scatters = []
        for link in self.link_value_dict.keys():
            voxels = self.get_voxels(link)
            if len(voxels) > 0 and link != 'Unoccupied':
                voxel_centers = np.asarray([voxel for voxel in voxels])
                voxel_size = self.voxel_size
                voxel_colors = [self.color_list[list(self.link_value_dict.keys()).index(link) % len(self.color_list)] for _ in range(len(voxels))]

                voxel_scatter = go.Scatter3d(
                    x=voxel_centers[:, 0],
                    y=voxel_centers[:, 1],
                    z=voxel_centers[:, 2],
                    mode='markers',
                    marker=dict(
                        size=voxel_size * 4,
                        color=voxel_colors
                    )
                )
                scatters.append(voxel_scatter)
        fig = go.Figure(data=scatters+[mesh_plotly] if mesh_plotly is not None else scatters)
        fig.update_layout(
            autosize=False,
            margin = {'l':0,'r':0,'t':0,'b':0},
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)", 
                        zeroline=False, showbackground=False, title=''),  # Remove X-axis title
                yaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
                        zeroline=False, showbackground=False, title=''),  # Remove Y-axis title
                zaxis=dict(showgrid=False, showticklabels=False, backgroundcolor="rgba(0,0,0,0)",
                        zeroline=False, showbackground=False, title=''),  # Remove Z-axis title
            ),
            scene_aspectmode='data',
            # plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to be transparent
            # paper_bgcolor='rgba(0,0,0,0)',  # Set paper background to be transparent
            showlegend=False,  # Hide the legend
            annotations=[],  # Remove annotations
            scene_camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=-0.22, z=0), eye=dict(x=1.1, y=-0.9, z=0.4)),  # Optional: Adjust camera for better view
            width=740,
            height=600
        )
        fig.show()

class Mesh_Decomp:
    def __init__(self, args, mesh_loader: Mesh_Loader):
        self.args = args
        self.mesh = mesh_loader.scaled_mesh
        self.joint_dict = mesh_loader.scaled_joint_dict
        self.link_tree = mesh_loader.link_tree
        self.mesh_group = Mesh_Group(args)
        self.color_list = ["#2DB3F0", "#124860", "#8E75AF", "#2b2335", "#C03027", "#4d1310", "#a6caa1", "#748d71", 
                           "grey", "black", "blue", "green", "red", "yellow", "orange", "purple", "pink", 
                           "brown", "cyan", "magenta", "olive", "teal", "navy", "maroon", "lime", "aqua", "fuchsia", 
                           "silver", "gray"]
    
    def voxelization(self):
        """
        Voxelization the mesh data.
        """
        aabb = self.mesh.mesh_o3d.get_axis_aligned_bounding_box()
        mesh_tri = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh.mesh_o3d)
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_tri)
        min_bound = aabb.min_bound
        max_bound = aabb.max_bound
        x_range = np.arange(min_bound[0], max_bound[0], self.args.voxel_size)
        y_range = np.arange(min_bound[1], max_bound[1], self.args.voxel_size)
        z_range = np.arange(min_bound[2], max_bound[2], self.args.voxel_size)
        query_points = np.stack(np.meshgrid(x_range, y_range, z_range, indexing='ij'), axis=-1).astype(np.float32)
        points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)
        occupancy_result = scene.compute_occupancy(points_tensor).numpy()
        erode_occupancy_result = occupancy_result

        # plt.imshow(erode_occupancy_result[:, :, 55])
        # plt.show()

        occupied_voxels = np.argwhere(erode_occupancy_result > 0)
        unoccupied_voxels = np.argwhere(erode_occupancy_result <= 0)

        self.occupied_voxels = np.array([x_range[occupied_voxels[:, 0]], 
                                         y_range[occupied_voxels[:, 1]], 
                                         z_range[occupied_voxels[:, 2]]]).T
        self.unoccupied_voxels = np.array(
            [x_range[unoccupied_voxels[:,0]],
            y_range[unoccupied_voxels[:,1]],
            z_range[unoccupied_voxels[:,2]]]
        ).T

        self.mesh_group.set_range(x_range, y_range, z_range)
        self.mesh_group.set_voxels('BODY', self.occupied_voxels)

        return self.occupied_voxels, self.unoccupied_voxels
    


    def decompose(self):
        """
        Decompose the mesh data into different parts according to the joints.
        """
        self.voxelization()
        self.father_link_dict = {}
        cur_node = self.link_tree
        self.decompose_result = np.array(['UNCLASSIFIED'] * len(self.occupied_voxels), dtype=object)

        # Collect all joints and links
        all_joints = {}
        all_links = {}
        queue = [cur_node]
        while queue:
            node = queue.pop(0)
            
            all_links[node.val.name] = node.val
            for joint_name, joint_pos in node.val.joints.items():
                if joint_name not in all_joints:
                    all_joints[joint_name] = []
                all_joints[joint_name].append((node.val.name, joint_pos))
            queue.extend(node.children)

            # Fill in father link dict for later use
            child_nodes = node.children
            for child_node in child_nodes:
                self.father_link_dict[child_node.val.name] = node.val
        

        # Create line segments for each link
        link_segments = {}
        for link_name, link in all_links.items():
            segments = []
            joint_positions = list(link.joints.values())
            for i in range(len(joint_positions)):
                for j in range(i+1, len(joint_positions)):
                    segments.append((joint_positions[i], joint_positions[j]))
            link_segments[link_name] = segments

        # Function to calculate distance from point to line segment
        def point_to_segment_distance(point, segment):
            p1, p2 = segment
            line_vec = p2 - p1
            point_vec = point - p1
            line_len = np.linalg.norm(line_vec)
            if line_len == 0:
                return np.linalg.norm(point_vec)
            t = max(0, min(1, np.dot(point_vec, line_vec) / (line_len * line_len)))
            projection = p1 + t * line_vec
            return np.linalg.norm(point - projection)

        # Iterate through all voxels
        import tqdm
        for i in tqdm.tqdm(range(len(self.occupied_voxels))):
            voxel = self.occupied_voxels[i]
            closest_joint = None
            min_distance = float('inf')
            
            # Find the closest joint
            for joint_name, joint_info in all_joints.items():
                for link_name, joint_pos in joint_info:
                    distance = np.linalg.norm(voxel - joint_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_joint = joint_name

            # Determine which link the voxel belongs to
            if len(all_joints[closest_joint]) == 1:
                # If the joint belongs to only one link, assign the voxel to that link
                self.decompose_result[i] = all_joints[closest_joint][0][0]
            else:
                # If the joint belongs to multiple links, find the closest line segment
                min_segment_distance = float('inf')
                closest_link = None
                for link_name, _ in all_joints[closest_joint]:
                    for segment in link_segments[link_name]:
                        segment_distance = point_to_segment_distance(voxel, segment)
                        if segment_distance < min_segment_distance:
                            min_segment_distance = segment_distance
                            closest_link = link_name
                self.decompose_result[i] = closest_link

        #  Update mesh_group with decomposed voxels and perform clustering for each link
        for link_name in tqdm.tqdm(np.unique(self.decompose_result)):
            if link_name != 'UNCLASSIFIED':
                link_voxels = self.occupied_voxels[self.decompose_result == link_name]
                
                # Perform DBSCAN clustering for this link's voxels
                epsilon = self.args.voxel_size * 1.732
                min_samples = 5  # Minimum number of samples in a cluster
                clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(link_voxels)
                
                # Find the largest cluster for this link
                labels = clustering.labels_

                if len(np.unique(labels)) > 1:  # If there's more than one cluster (including noise)
                    unique_labels, label_counts = np.unique(labels[labels != -1], return_counts=True)
                    largest_cluster = unique_labels[np.argmax(label_counts)]
                    mask = labels == largest_cluster
                    largest_cluster_voxels = link_voxels[mask]
                    
                    # # Update decompose_result for this link
                    # link_indices = np.where(self.decompose_result == link_name)[0]
                    # self.decompose_result[link_indices[~mask]] = "UNCLASSIFIED"
                    
                    # Update mesh_group with the largest cluster of voxels for this link
                    self.mesh_group.set_voxels(link_name, largest_cluster_voxels)

                    # Make the rest of the voxels unoccupied
                    unoccupied_voxels = link_voxels[~mask]
                    self.mesh_group.set_voxels("Unoccupied", unoccupied_voxels)
                else:
                    # If there's only one cluster, keep all voxels
                    self.mesh_group.set_voxels(link_name, link_voxels)
        
        # Print the number of voxels for each link
        print("Number of clusters: ", len(self.mesh_group.link_value_dict.keys()))
        for link_name in self.mesh_group.link_value_dict.keys():
            print(f"Link {link_name}: {self.mesh_group.get_voxels(link_name).shape[0]} voxels")

        return self.mesh_group


    # def decompose_old(self):
    #     """
    #     Decompose the mesh data into different parts according to the joints.
    #     """
    #     self.voxelization()
    #     self.father_link_dict = {}
    #     cur_node = self.link_tree
    #     self.decompose_result = np.array(['UNCLASSIFIED'] * len(self.occupied_voxels), dtype=object)

    #     # Implement BFS to decompose the mesh data
    #     queue = [cur_node]
    #     while queue:
    #         root_node = queue.pop(0)
    #         child_nodes = root_node.children
    #         if not child_nodes:
    #             continue
            
    #         ## Cluster the voxels according to joint clusters
    #         joint_cluster = []
    #         name_cluster = []
    #         link_dict = {}
    #         joint_cluster.append(np.array(list(root_node.val.joints.values())))
    #         name_cluster.append(root_node.val.name)
    #         link_dict[root_node.val.name] = root_node.val
    #         for child_node in child_nodes:
    #             link_dict[child_node.val.name] = child_node.val
    #             self.father_link_dict[child_node.val.name] = root_node.val
    #             all_child_nodes, _ = child_node.get_all_children()
    #             all_child_nodes.append(child_node)
    #             child_joint_cluster = np.vstack([np.array(list(node.val.joints.values())) for node in all_child_nodes])
    #             joint_cluster.append(np.array(child_joint_cluster))
    #             name_cluster.append(child_node.val.name)
    #             queue.append(child_node)
            
    #         ## Cluster the voxels according to joint clusters
    #         distances = [np.min(np.linalg.norm(joint_cluster[i] - (np.repeat(self.mesh_group.get_voxels(root_node.val.name), joint_cluster[i].shape[0], axis=0).reshape(-1, joint_cluster[i].shape[0], 3)), axis=-1), axis=1) for i in range(len(joint_cluster))]
    #         distances = np.array(distances).T
    #         min_indices = [np.argwhere(distances[i] == np.min(distances[i])) for i in range(len(distances))]

    #         ## If there are multiple minimum values, choose the one with the largest distance to the nearest joint line
    #         min_indice = []
    #         for i in tqdm.tqdm(range(len(min_indices))):
    #             if min_indices[i].shape != (1,1):
    #                 link1 = name_cluster[min_indices[i][0][0]]
    #                 link2 = name_cluster[min_indices[i][1][0]]
    #                 voxel_pos = self.mesh_group.get_voxels(root_node.val.name)[i]
    #                 if link_dict[link1].get_min_axis_distance(voxel_pos) > link_dict[link2].get_min_axis_distance(voxel_pos):
    #                     min_indice.append(min_indices[i][1][0])
    #                 else:
    #                     min_indice.append(min_indices[i][0][0])
    #             else:
    #                 min_indice.append(min_indices[i][0][0])

    #         name_cluster = np.array(name_cluster)
    #         self.decompose_result = name_cluster[min_indice]

    #         clustered_voxels = {}
    #         for link_name in np.unique(self.decompose_result):
    #             if link_name not in self.mesh_group.link_value_dict.keys():
    #                 clustered_voxels[link_name] = self.mesh_group.get_voxels(root_node.val.name)[self.decompose_result == link_name]
    #         clustered_voxels[root_node.val.name] = self.mesh_group.get_voxels(root_node.val.name)[self.decompose_result == root_node.val.name]

    #         for link_name in np.unique(self.decompose_result):
    #             if link_name not in self.mesh_group.link_value_dict.keys():
    #                 self.mesh_group.set_voxels(link_name, clustered_voxels[link_name])
    #         self.mesh_group.set_voxels(root_node.val.name, clustered_voxels[root_node.val.name])
        
    #     # Only preserve the largest cluster for each link
    #     unique_types = np.unique(self.mesh_group.voxel_data)
        
    #     for voxel_type in unique_types:
    #         if voxel_type == 0:  # Assuming 0 might be used for background or empty space
    #             continue
            
    #         # Create a binary volume for the current type
    #         binary_volume = (self.mesh_group.voxel_data == voxel_type)
    #         structure = np.zeros((3, 3, 3))
    #         structure[1, 1, :] = 1
    #         structure[1, :, 1] = 1
    #         structure[:, 1, 1] = 1
    #         labeled_volume, _ = label(binary_volume, structure=structure)
    #         cluster_sizes = np.bincount(labeled_volume.ravel())[1:] 
    #         if cluster_sizes.size == 0:
    #             continue  # No clusters of this type
    #         largest_cluster_idx = cluster_sizes.argmax() + 1  # +1 because bincount skips the background label 0
    #         self.mesh_group.voxel_data[(labeled_volume != largest_cluster_idx) & (labeled_volume != 0)] = 0
        
    #     # Print the number of voxels for each link
    #     print("Number of clusters: ", len(self.mesh_group.link_value_dict.keys()))
    #     for link_name in self.mesh_group.link_value_dict.keys():
    #         print(f"Link {link_name}: {self.mesh_group.get_voxels(link_name).shape[0]} voxels")

    #     return self.mesh_group

    def generate_ideal_urdf(self):
        """
        Generate the URDF file.
        """
        dir = './urdf/' + self.args.model_name + '/tmp/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        self.temp_dir = dir
        urdf_file = open(dir + self.args.model_name + '_ideal.urdf', 'w+')

        self.urdf_dir = dir + self.args.model_name + '_ideal.urdf'
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

            # Write current link
            cur_link = root_node.val
            link_dir = dir + cur_link.name + '_ideal' + '.stl'
            voxel_grid_to_mesh(voxel_positions=self.mesh_group.get_voxels(cur_link.name), dir=link_dir, voxel_size=self.args.voxel_size)

            rel_pos = (np.array(cur_link.axis[0]) - np.array(self.father_link_dict[cur_link.name].axis[0]) if cur_link.name != "BODY" else np.array(cur_link.axis[0])) / 100.0
            link_visual = {
                "origin": {"xyz": ' '.join(map(str, -np.array(cur_link.axis[0]) / 100.0)), "rpy": '0 0 0'},
                "geometry": {"filename": "package://anything2robot/urdf/" + self.args.model_name + "/tmp/" + cur_link.name + "_ideal.stl"},
                "material": "grey"
            }
            link_collision = {
                "origin": {"xyz": ' '.join(map(str, -np.array(cur_link.axis[0]) / 100.0)), "rpy": '0 0 0'},
                "geometry": {"filename": "package://anything2robot/urdf/" + self.args.model_name + "/tmp/" + cur_link.name + "_ideal.stl"}
            }
            per_voxel_mass = self.args.voxel_density * (self.args.voxel_size ** 3)
            part_mass = per_voxel_mass * self.mesh_group.get_voxels(cur_link.name).shape[0]
            inetial_matrix, CoM = calculate_inertia_tensor(self.mesh_group.get_voxels(cur_link.name) / 100.0, part_mass, np.eye(4))
            link_inertial = {
                "origin": {"xyz": ' '.join(map(str, CoM + (rel_pos - np.array(cur_link.axis[0])) / 100.0)), "rpy": '0 0 0'},
                "mass": str(part_mass),
                "inertia": {"ixx": inetial_matrix[0, 0], "iyy": inetial_matrix[1, 1], "izz": inetial_matrix[2, 2], "ixy": inetial_matrix[0, 1], "ixz": inetial_matrix[0, 2], "iyz": inetial_matrix[1, 2]}
            }
            write_link(urdf_file=urdf_file, link_name=cur_link.name, visual=link_visual, collision=link_collision, inertial=link_inertial)
            
            self.ideal_mass += part_mass

            if cur_link.name != "BODY":
                # Write a 2DoF joint
                if len(cur_link.axis) == 3:
                    write_link(urdf_file=urdf_file, link_name=cur_link.name + '_virtual')
                    joint1 = {
                        "joint_name": cur_link.name + '_joint1',
                        "joint_type": "revolute",
                        "parent_link": self.father_link_dict[cur_link.name].name,
                        "child_link": cur_link.name + '_virtual',
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, cur_link.axis[1]))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **joint1)
                    joint2 = {
                        "joint_name": cur_link.name + '_joint2',
                        "joint_type": "revolute",
                        "parent_link": cur_link.name + '_virtual',
                        "child_link": cur_link.name,
                        "origin": {"xyz": ' '.join(map(str, np.zeros(3))), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, cur_link.axis[2]))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **joint2)
                elif len(cur_link.axis) == 2:
                    cur_joint = {
                        "joint_name": cur_link.name + '_joint',
                        "joint_type": "revolute",
                        "parent_link": self.father_link_dict[cur_link.name].name,
                        "child_link": cur_link.name,
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": ' '.join(map(str, cur_link.axis[1]))},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "20", "velocity": "1.5"}
                    }
                    write_joint(urdf_file, **cur_joint)
        
        # Writre foot link
        def contact_link_func(link):
            for joint_name in link.joints.keys():
                if 'foot' in joint_name:
                    return True
            return False
        contact_nodes = self.link_tree.find_children(contact_link_func, None)
        for contact_node in contact_nodes:
            for joint_name, joint_pos in contact_node.val.joints.items():
                if 'foot' in joint_name:
                    write_link(urdf_file=urdf_file, link_name=joint_name)
                    rel_pos = np.array(np.array(joint_pos) - contact_node.val.axis[0]) / 100.0
                    foot_joint = {
                        "joint_name": joint_name + '_joint',
                        "joint_type": "revolute",
                        "parent_link": contact_node.val.name,
                        "child_link": joint_name,
                        "origin": {"xyz": ' '.join(map(str, rel_pos)), "rpy": "0 0 0"},
                        "axis": {"xyz": "1 0 0"},
                        "limit": {"lower": "-1.57", "upper": "1.57", "effort": "0", "velocity": "0"}
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

        print("ideal mass:", self.ideal_mass)
    
    def generate_constraints(self):
        """
        Generate constraints by sampling in configuration space
        """
        self.generate_ideal_urdf()
        urdf_dir = self.urdf_dir
        pkg_dir = './..'

        model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_dir, pkg_dir)

        # Build a data frame associated with the model
        data = model.createData()
        max_torque = np.zeros((model.nv))

        # Contact force that will be applied to foot links
        force = np.array([0, 0, 9.81 * self.ideal_mass / 2])  #CHG.

        # Sample a random joint configuration, joint velocities and accelerations
        for i in range(100):
            q = pin.randomConfiguration(model)
            v = np.zeros((model.nv, 1))  # in rad/s 
            a = np.zeros((model.nv, 1))  # in rad/s² 
            
            # Add external forces to the link which contains the joint whose name contains 'foot'    
            fs_ext = [pin.Force(np.array([0,0,0,0,0,0])) for _ in range(len(model.joints))]
            for i, joint_name in enumerate(model.names):
                if 'foot' in joint_name:
                    fs_ext[i] = pin.Force(np.hstack((force, np.zeros(3))))
            tau = pin.rnea(model, data, q, v, a, fs_ext)
            max_torque = np.maximum(max_torque, np.abs(tau))

        # Get joint names
        joint_names = []
        result_torque = []
        for i, joint_name in enumerate(model.names):
            if (joint_name != 'universe') and ('foot' not in joint_name):
                joint_names.append(joint_name)
                result_torque.append(np.round(max_torque[i-1], 2))
        return joint_names, result_torque

    def render(self):
        """
        Render the decomposed voxels.
        """
        self.mesh_group.render(self.mesh.mesh_plotly)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mesh Loader')
    parser.add_argument('--model_name', type=str, default='lynel', help='The model name')
    parser.add_argument('--expected_x', type=float, default=40, help='The expected width of the model')
    parser.add_argument('--voxel_size', type=float, default=0.5, help='The size of the voxel')
    parser.add_argument('--voxel_density', type=float, default=2e-4, help='The density of the voxel')
    args = parser.parse_args()
    mesh_loader = Custom_Mesh_Loader(args)
    mesh_dir = os.path.normpath('./model/given_models/' + args.model_name + '.stl')
    joint_dir = os.path.normpath('./model/given_models/' + args.model_name + '_joints.pkl')
    mesh_loader.load_mesh(mesh_dir)
    mesh_loader.load_joint_positions(joint_dir)
    mesh_loader.scale()
    # mesh_loader.render()
    
    mesh_decomp = Mesh_Decomp(args, mesh_loader)
    mesh_decomp.decompose()
    # joint_names, max_torque = mesh_decomp.generate_constraints()
    # for joint_name, torque in zip(joint_names, max_torque):
    #     print(f'{joint_name}: {torque}')
    mesh_decomp.render()

    # # use plotly to render the 3D array valued with int
    # def render_voxels(data):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')

    #     # Get the indices where the voxel value is greater than zero
    #     x, y, z = np.where(data > 0)

    #     # Plot these points
    #     ax.scatter(x, y, z, zdir='z', c='red', marker='o')

    #     ax.set_xlabel('X Dimension')
    #     ax.set_ylabel('Y Dimension')
    #     ax.set_zlabel('Z Dimension')
    #     ax.set_title('3D Visualization of Voxels > 0')

    #     # Show the plot
    #     plt.show()

    # # Call the function with your data
    # render_voxels(mesh_decomp.mesh_group.voxel_data)
    
