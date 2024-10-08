import pickle as pkl
import numpy as np
import trimesh
import open3d as o3d
import sys
import os

# Get the path of this script
script_path = os.path.dirname(os.path.realpath(__file__))
print("Script path: ", script_path)

#sys.path.append('/home/clarence/ros_ws/metamaterial_ws/src/auto_design/modules')

sys.path.append(os.path.join(script_path, '../../../auto_design/modules'))
sys.path.append(os.path.join(script_path, '../../../auto_design'))
from interference_removal import RobotOptResult, LinkResult



def transform_trimesh(mesh, transformation_matrix, save_path=None):
    # Apply transformation
    mesh.apply_transform(transformation_matrix)

    if save_path is not None:
        mesh.export(save_path)

    return mesh


def load_and_transform_stl(file_path, transformation_matrix, scale=1.0, save_path=None):
    # Load STL file using trimesh
    mesh = trimesh.load(file_path)
    # Scale mesh
    mesh.apply_scale(scale)
    # Apply transformation
    mesh.apply_transform(transformation_matrix)

    if save_path is not None:
        mesh.export(save_path)

    return mesh

def convert_to_open3d(trimesh_mesh):
    vertices = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    open3d_mesh.compute_vertex_normals()
    return open3d_mesh


def visualize_meshes(stl_files, transformation_matrices, scales=None):
    open3d_meshes = []
    for i in range(len(stl_files)):
        if scales is not None:
            mesh = load_and_transform_stl(stl_files[i], transformation_matrices[i], scales[i])
        else:
            mesh = load_and_transform_stl(stl_files[i], transformation_matrices[i])
        open3d_mesh = convert_to_open3d(mesh)
        open3d_meshes.append(open3d_mesh)

    # Create coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    open3d_meshes.append(coordinate_frame)

    o3d.visualization.draw_geometries(open3d_meshes)


def get_rotation_matrix(v1, v2):
    # CHECK IF v1 and v2 are parallel
    if np.allclose(v1, v2):
        return np.eye(3)
    
    if np.allclose(v1, -v2): # Avoid the singularity when v1 and v2 are opposite. Modify the v2 to avoid singularity
        v2 = np.array([0.001, 1, 0.001])

    # Normalize the vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Compute the cross product (rotation axis)
    cross_prod = np.cross(v1, v2)
    
    # Compute the dot product (cosine of the rotation angle)
    dot_prod = np.dot(v1, v2)
    
    # Handle the special cases when v1 and v2 are parallel or opposite
    if np.allclose(cross_prod, [0, 0, 0]):
        if dot_prod > 0:
            return np.eye(3)  # No rotation needed
        else:
            # Rotate 180 degrees around any orthogonal axis
            orthogonal_axis = np.array([1, 0, 0]) if not np.allclose(v1, [1, 0, 0]) else np.array([0, 1, 0])
            cross_prod = np.cross(v1, orthogonal_axis)
    
    # Skew-symmetric cross-product matrix
    K = np.array([
        [0, -cross_prod[2], cross_prod[1]],
        [cross_prod[2], 0, -cross_prod[0]],
        [-cross_prod[1], cross_prod[0], 0]
    ])
    
    # Rotation matrix using Rodrigues' rotation formula
    I = np.eye(3)
    R = I + K + K @ K * ((1 - dot_prod) / (np.linalg.norm(cross_prod) ** 2))
    
    return R



def get_rotation_matrix_from_angle(axis, angle):
    """
    Returns the rotation matrix for a rotation around a given axis by a specified angle.
    
    Parameters:
    axis (numpy array): The axis of rotation (must be a unit vector).
    angle (float): The rotation angle in radians.
    
    Returns:
    numpy array: The rotation matrix (3x3).
    """
    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)
    
    # Calculate trigonometric functions
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    one_minus_cos = 1 - cos_angle
    
    # Components of the axis
    x, y, z = axis
    
    # Rotation matrix using the Rodrigues' rotation formula
    R = np.array([
        [cos_angle + x**2 * one_minus_cos,       x*y*one_minus_cos - z*sin_angle, x*z*one_minus_cos + y*sin_angle],
        [y*x*one_minus_cos + z*sin_angle, cos_angle + y**2 * one_minus_cos,       y*z*one_minus_cos - x*sin_angle],
        [z*x*one_minus_cos - y*sin_angle, z*y*one_minus_cos + x*sin_angle, cos_angle + z**2 * one_minus_cos]
    ])
    
    return R

def get_transformation_matrix_from_angle(rotation_point, axis, angle):
    """
    Returns the transformation matrix for a rotation around a given axis and a specific point.
    
    Parameters:
    rotation_point (numpy array): The point around which to rotate.
    axis (numpy array): The axis of rotation.
    angle (float): The rotation angle in radians.
    
    Returns:
    numpy array: The 4x4 transformation matrix.
    """
    # Translate the point of rotation to the origin
    T1 = np.eye(4)
    T1[:3, 3] = -rotation_point

    # Rotation matrix
    R = get_rotation_matrix_from_angle(axis, angle)
    
    # 4x4 rotation matrix
    R4 = np.eye(4)
    R4[:3, :3] = R
    
    # Translate back to the original point
    T2 = np.eye(4)
    T2[:3, 3] = rotation_point

    # Combine transformations
    transformation_matrix = T2 @ R4 @ T1
    
    return transformation_matrix



if __name__ == "__main__":
    # Load robot result
    robot_result = pkl.load(open('/home/clarence/ros_ws/metamaterial_ws/src/auto_design/results/lynel_robot_result.pkl', 'rb'))
    #robot_result = pkl.load(open('/home/clarence/git/anything2robot/anything2robot/auto_design/modules/lynel_robot_result.pkl', 'rb'))

    FR_LOW_link = robot_result.link_dict['FR_LOW']

    # for link_name in robot_result.link_dict:
    #     print("Link name: ", link_name)
    #     print("Link Tenon Positions: ", robot_result.link_dict[link_name].tenon_pos)
    #     print("Link Torques: ", robot_result.link_dict[link_name].applied_torque)
    #     print("Link tenon_type: ", robot_result.link_dict[link_name].tenon_type)

    # exit()

    stl_file_path = '/home/clarence/ros_ws/metamaterial_ws/src/metamaterial_filling/data/lynel/20240721/tmp/FR_LOW.stl'
    tenon_stl = '/home/clarence/git/anything2robot/anything2robot/metamaterial_filling/tenon/connection_child.stl'

    eye_transformation_matrix = np.eye(4)

    # Rotate 90 degrees around x-axis
    tenon_transformation_matrix = np.array([[1, 0, 0, 0.0],
                                            [0, 0, -1, 0.0],
                                            [0, 1, 0, 0.0],
                                            [0, 0, 0, 1]])

    tenon_direction_vector = np.array([0, -1, 0])
    FR_LOW_tenon_pos_direction_vector = np.array([FR_LOW_link.tenon_pos[0][3], FR_LOW_link.tenon_pos[0][4], FR_LOW_link.tenon_pos[0][5]])

    rotation_matrix = get_rotation_matrix(tenon_direction_vector, FR_LOW_tenon_pos_direction_vector)

    transformation_to_FR_LOW = np.array([[rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2], FR_LOW_link.tenon_pos[0][0]],
                                            [rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2], FR_LOW_link.tenon_pos[0][1]],
                                            [rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2], FR_LOW_link.tenon_pos[0][2]],
                                            [0, 0, 0, 1]])

    tenon_transformation_matrix = np.dot(transformation_to_FR_LOW, tenon_transformation_matrix)


    stl_files = [stl_file_path, tenon_stl]
    scales = [1.0, 0.001]
    transformation_matrices = [eye_transformation_matrix, tenon_transformation_matrix]

    visualize_meshes(stl_files, transformation_matrices, scales)


