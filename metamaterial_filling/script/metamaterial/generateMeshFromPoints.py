
from readInnerPointsBin import read_inner_points_bin
import numpy as np
import argparse
import os

from skimage.measure import marching_cubes
from math import pi
import open3d as o3d

def voxel_grid_to_mesh(voxel_positions, voxel_size, stl_save_apth, output=True):
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

    # Initialize the voxel map
    voxel_map = np.ones(dimensions, dtype=int)

    # Mark occupied voxels
    for position in voxel_positions:
        indices = np.ceil((position - min_bounds) / voxel_size).astype(int)
        voxel_map[tuple(indices)] = 0
        indices = np.floor((position - min_bounds) / voxel_size).astype(int)
        voxel_map[tuple(indices)] = 0

    # Apply Marching Cubes
    verts, faces, _, _ = marching_cubes(voxel_map)

    # Scale vertices according to the voxel size
    verts *= voxel_size

    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Segment the mesh into connected components
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
        # mesh.scale(1/100, center=(0,0,0))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(stl_save_apth, mesh)

    return mesh

def generate_mesh_from_points(points_bin_file, voxel_size, stl_save_apth):
    # Read the inner points
    points = read_inner_points_bin(points_bin_file)

    # Convert the points to a mesh
    voxel_grid_to_mesh(points, voxel_size, stl_save_apth)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Generate a mesh from points')

    arg_parser.add_argument('--points_bin_file', type=str, default='data/output/FL_replaced_small_points.bin', help='The path to the binary file containing the inner points')
    arg_parser.add_argument('--voxel_size', type=float, default=1, help='The size of each voxel. mm')
    arg_parser.add_argument('--stl_save_apth', type=str, default='data/output/FL_replaced_small_mesh.stl', help='The path to save the mesh')

    args = arg_parser.parse_args()

    generate_mesh_from_points(args.points_bin_file, args.voxel_size, args.stl_save_apth)
    