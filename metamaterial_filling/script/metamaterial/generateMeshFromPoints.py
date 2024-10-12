
import numpy as np
import argparse
import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_dir + '/metamaterial_filling/script')
sys.path.append(project_dir + '/metamaterial_filling/script/metamaterial')

from skimage.measure import marching_cubes
from math import pi
import open3d as o3d

from readInnerPointsBin import read_inner_points_bin
from sklearn.cluster import DBSCAN
import trimesh
import matplotlib.pyplot as plt


from collections import deque

def custom_flood_fill(voxel_map, start_pos, fill_value):
    # Copy the original voxel map to avoid modifying it in place
    filled_voxel_map = np.copy(voxel_map)

    # Define the 6 possible neighbor directions (up, down, left, right, forward, backward)
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # Initialize a queue for BFS and add the starting position
    queue = deque([start_pos])

    # Perform BFS flood fill
    while queue:
        x, y, z = queue.popleft()
        
        # If the current voxel is already filled, continue
        if filled_voxel_map[x, y, z] == fill_value:
            continue
        
        # Fill the current voxel
        filled_voxel_map[x, y, z] = fill_value
        
        # Check all 6 neighbors
        for dx, dy, dz in directions:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < filled_voxel_map.shape[0] and \
               0 <= ny < filled_voxel_map.shape[1] and \
               0 <= nz < filled_voxel_map.shape[2] and \
               filled_voxel_map[nx, ny, nz] == 1:  # Only fill unoccupied voxels
                queue.append((nx, ny, nz))

    return filled_voxel_map

def fill_internal_holes(voxel_map):
    # Create a padded voxel map to handle boundary issues
    padded_voxel_map = np.pad(voxel_map, pad_width=1, mode='constant', constant_values=1)

    # Start flood fill from a corner (which should be external space)
    filled_from_exterior = custom_flood_fill(padded_voxel_map, (0, 0, 0), fill_value=2)

    # Remove the padding
    filled_from_exterior = filled_from_exterior[1:-1, 1:-1, 1:-1]

    # Fill internal holes (any remaining 1s)
    filled_voxel_map = np.where(filled_from_exterior == 1, 0, voxel_map)

    return filled_voxel_map


def voxel_grid_to_mesh(voxel_positions, voxel_size, stl_save_apth, output=True, draw_points=False):
    """
    Convert a voxel grid to a mesh using the Marching Cubes algorithm.

    :param voxel_grid: Open3D VoxelGrid object.
    :param voxel_size: Size of each voxel.
    :return: Open3D TriangleMesh object.
    """
    
    # Calculate the dimensions of the voxel map
    min_bounds = np.min(voxel_positions, axis=0) - 2*voxel_size
    max_bounds = np.max(voxel_positions, axis=0) + 2*voxel_size
    dimensions = np.ceil((max_bounds - min_bounds) / voxel_size).astype(int)

    print("min_bounds: ", min_bounds)
    print("max_bounds: ", max_bounds)
    print("dimensions: ", dimensions)
    print("voxel_size: ", voxel_size)

    # Initialize the voxel map
    voxel_map = np.ones(dimensions, dtype=int)

    # Mark occupied voxels
    for position in voxel_positions:
        indices = np.ceil((position - min_bounds) / voxel_size).astype(int)
        voxel_map[tuple(indices)] = 0
        indices = np.floor((position - min_bounds) / voxel_size).astype(int)
        voxel_map[tuple(indices)] = 0
    
    voxel_map = fill_internal_holes(voxel_map)

    # Draw the points whose value is 0 in the voxel map
    if draw_points:
        points = np.argwhere(voxel_map == 0)
        points = points * voxel_size + min_bounds
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])


    # Apply Marching Cubes
    verts, faces, _, _ = marching_cubes(voxel_map)

    # Scale vertices according to the voxel size
    verts *= voxel_size

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # # Segment the mesh into connected components
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Find the index of the largest cluster
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < np.max(cluster_n_triangles)
    mesh.remove_triangles_by_mask(triangles_to_remove)
    
    # Translate the mesh to the origin
    mesh = mesh.translate(min_bounds)
    mesh = mesh.filter_smooth_laplacian(3)

    if output:
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(stl_save_apth, mesh)

    return mesh

def generate_mesh_from_points(points_bin_file, voxel_size, stl_save_apth, draw_points=False):
    # Read the inner points
    points = read_inner_points_bin(points_bin_file)

    # Get the largest cluster
    points = largest_cluster(points, voxel_size)

    # Keep only the points whose x > 0
    #points = points[points[:, 1] < -50]
    
    # Draw the points
    if draw_points:
        points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()


    # Convert the points to a mesh
    voxel_grid_to_mesh(points, voxel_size, stl_save_apth, draw_points=draw_points)


def largest_cluster(points, voxel_size):
    print("Finding the largest cluster...")
    # Step 1: Apply DBSCAN
    dbscan = DBSCAN(eps=voxel_size*1.732, min_samples=5) # eps: Maximum distance between two samples for them to be considered as in the same neighborhood,  min_samples: Minimum number of points to form a cluster
    labels = dbscan.fit_predict(points)

    # Step 2: Identify the largest cluster
    # Cluster labels -1 means noise (outliers), so we exclude those
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) > 0:
        largest_cluster_label = unique_labels[np.argmax(counts)]
    else:
        raise ValueError("No clusters found")

    # Step 3: Select only the points from the largest cluster
    largest_cluster_points = points[labels == largest_cluster_label]

    # Output the largest cluster points
    print("Largest cluster points:")
    print(largest_cluster_points) 

    return largest_cluster_points


if __name__ == '__main__':
    # mesh_to_check = trimesh.load('data/output/BODY_UP_replaced_smaller.stl')
    # print("Is the mesh watertight: ", mesh_to_check.is_watertight)
    # exit()

    arg_parser = argparse.ArgumentParser(description='Generate a mesh from points')

    arg_parser.add_argument('--points_bin_file', type=str, default='data/output/BODY_replaced_smaller_points.bin', help='The path to the binary file containing the inner points')
    arg_parser.add_argument('--voxel_size', type=float, default=2, help='The size of each voxel. mm')
    arg_parser.add_argument('--stl_save_apth', type=str, default='data/output/BODY_replaced_smaller.stl', help='The path to save the mesh')

    args = arg_parser.parse_args()

    generate_mesh_from_points(args.points_bin_file, args.voxel_size, args.stl_save_apth, draw_points=True)

    # Load the model and check if the mesh is watertight
    mesh = o3d.io.read_triangle_mesh(args.stl_save_apth)
    print("Is the mesh watertight: ", mesh.is_watertight)

    mesh_to_check = trimesh.load(args.stl_save_apth)
    print("Is the mesh watertight: ", mesh_to_check.is_watertight)

    