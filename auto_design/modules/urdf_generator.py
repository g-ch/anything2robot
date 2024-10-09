import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes
from math import pi

def voxel_grid_to_mesh(voxel_positions, dir, voxel_size, output=True):
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
        mesh.scale(1/100, center=(0,0,0))
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(dir, mesh)

    return mesh

def matrix_to_xyz_rpy(matrix):
    # Extract the translation (position)
    x, y, z = matrix[:3, 3]

    # Extract the rotation matrix
    R = matrix[:3, :3]

    # Calculate the roll, pitch, and yaw angles (XYZ rotation order)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0

    return [x, y, z, roll, pitch, yaw]

# calculate rpy angle from axis direction
def calculate_rpy(x_axis, y_axis, z_axis):
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    R = np.vstack([x_axis, y_axis, z_axis]).T
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]), R


apply_transform = lambda point, transformation: (np.hstack((np.array(point).reshape(-1, 3), np.ones((np.array(point).reshape(-1, 3).shape[0], 1)))) @ transformation.T)[:,:3]

def calculate_inertia_tensor(voxels, mass, H):
    """
    Calculate the inertia tensor of a rigid body represented by voxels.
    
    Args:
    - voxels (numpy array): Array of voxel coordinates in shape (n, 3).
    - mass (float): Mass of the body.
    - H (numpy array): Homogeneous transformation matrix (4x4).
    
    Returns:
    - numpy array: Inertia tensor (3x3 matrix).
    """
    # Apply the transformation to the voxel coordinates
    transformed_voxels = (H @ np.hstack((voxels, np.ones((voxels.shape[0], 1)))).T).T[:, :3]
    center_of_mass = np.mean(transformed_voxels, axis=0)
    inertia_tensor = np.zeros((3, 3))

    Rs = transformed_voxels - center_of_mass
    inertia_tensor = np.sum(np.sum(Rs**2, axis=1)[:, np.newaxis, np.newaxis] * np.eye(3) - np.einsum('ij,ik->ijk', Rs, Rs), axis=0) * mass / voxels.shape[0]

    return inertia_tensor, center_of_mass

def write_material(urdf_file, name, rgba):
    urdf_file.write(f'  <material name="{name}">\n')
    urdf_file.write(f'    <color rgba="{rgba}"/>\n')
    urdf_file.write('  </material>\n')

def write_link(urdf_file, link_name, visual=None, collision=None, inertial=None, motors=None):

    urdf_file.write(f'  <link name="{link_name}">\n')

    if visual:
        urdf_file.write('    <visual>\n')
        urdf_file.write(f'      <origin xyz="{visual["origin"]["xyz"]}" rpy="{visual["origin"]["rpy"]}"/>\n')
        urdf_file.write('      <geometry>\n')
        urdf_file.write(f'        <mesh filename="{visual["geometry"]["filename"]}"/>\n')
        urdf_file.write('      </geometry>\n')
        if "material" in visual:
            urdf_file.write(f'      <material name="{visual["material"]}"/>\n')
        urdf_file.write('    </visual>\n')
    
    if motors:
        for motor in motors:
            urdf_file.write('    <visual>\n')
            urdf_file.write(f'      <origin xyz="{motor["xyz"]}" rpy="{motor["rpy"]}"/>\n')
            urdf_file.write('      <geometry>\n')
            urdf_file.write(f'        <mesh filename="{motor["filename"]}" scale="0.001 0.001 0.001"/>\n')
            urdf_file.write('      </geometry>\n')
            urdf_file.write(f'     <material name="{motor["type"]}"/>\n')
            urdf_file.write('    </visual>\n')

    if collision:
        urdf_file.write('    <collision>\n')
        urdf_file.write(f'      <origin xyz="{visual["origin"]["xyz"]}" rpy="{visual["origin"]["rpy"]}"/>\n')
        urdf_file.write('      <geometry>\n')
        urdf_file.write(f'        <mesh filename="{visual["geometry"]["filename"]}"/>\n')
        urdf_file.write('      </geometry>\n')
        urdf_file.write('    </collision>\n')

    if inertial:
        urdf_file.write('    <inertial>\n')
        urdf_file.write(f'      <origin xyz="{inertial["origin"]["xyz"]}" rpy="{inertial["origin"]["rpy"]}"/>\n')
        urdf_file.write(f'      <mass value="{inertial["mass"]}"/>\n')
        inertia = inertial["inertia"]
        urdf_file.write(f'      <inertia ixx="{inertia["ixx"]}" ixy="{inertia["ixy"]}" ixz="{inertia["ixz"]}" iyy="{inertia["iyy"]}" iyz="{inertia["iyz"]}" izz="{inertia["izz"]}"/>\n')
        urdf_file.write('    </inertial>\n')

    urdf_file.write('  </link>\n')

def write_joint(urdf_file, joint_name, joint_type, parent_link, child_link, origin, axis, limit=None):
    urdf_file.write(f'  <joint name="{joint_name}" type="{joint_type}">\n')
    urdf_file.write(f'    <parent link="{parent_link}"/>\n')
    urdf_file.write(f'    <child link="{child_link}"/>\n')
    urdf_file.write(f'    <origin xyz="{origin["xyz"]}" rpy="{origin["rpy"]}"/>\n')
    if joint_type != 'fixed':
        urdf_file.write(f'    <axis xyz="{axis["xyz"]}"/>\n')
        if limit:
            urdf_file.write(f'    <limit lower="{limit["lower"]}" upper="{limit["upper"]}" effort="{limit["effort"]}" velocity="{limit["velocity"]}"/>\n')
    urdf_file.write(f'  </joint>\n')


def write_transmission(urdf_file, transmission_name, joint_name, actuator_name):
    urdf_file.write(f' <transmission name="{transmission_name}">\n')
    urdf_file.write('     <type>transmission_interface/SimpleTransmission</type>\n')
    urdf_file.write(f'     <joint name="{joint_name}">\n')
    urdf_file.write('     <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>\n')
    urdf_file.write('     </joint>\n')
    urdf_file.write(f'     <actuator name="{actuator_name}">\n')
    urdf_file.write('     <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>\n')
    urdf_file.write('     <mechanicalReduction>1</mechanicalReduction>\n')
    urdf_file.write('     </actuator>\n')
    urdf_file.write(' </transmission>\n')


def calculate_mass(shape, dimensions, density=1000):
    if shape == 'cylinder':
        radius, height = dimensions
        volume = pi * radius ** 2 * height
    elif shape == 'sphere':
        radius = dimensions[0]
        volume = 4/3 * pi * radius ** 3
    else:
        volume = 0  # Default case for unknown shapes
    return density * volume

def inertia_cylinder(mass, radius, height):
    I_xx = I_yy = 1/12 * mass * (3 * float(radius) ** 2 + float(height) ** 2)
    I_zz = 0.5 * mass * float(radius) ** 2
    return {"ixx": I_xx, "iyy": I_yy, "izz": I_zz, "ixy": 0, "ixz": 0, "iyz": 0}

def inertia_sphere(mass, radius):
    I = 2/5 * mass * float(radius) ** 2
    return {"ixx": I, "iyy": I, "izz": I, "ixy": 0, "ixz": 0, "iyz": 0}

def get_collision(cylinder_top, cylinder_bottom):
    rpy = calculate_rpy(np.cross(cylinder_top - cylinder_bottom, np.array([1, 1, 1])), np.cross(cylinder_top - cylinder_bottom, np.cross(cylinder_top - cylinder_bottom, np.array([1, 1, 1]))), cylinder_top - cylinder_bottom)[0]
    return {
        "origin": {"xyz": ' '.join(map(str, ((cylinder_top + cylinder_bottom) / 2).flatten())), "rpy": ' '.join(map(str, rpy))},
        "geometry": {"cylinder": {"length": str(np.linalg.norm(cylinder_top - cylinder_bottom)), "radius": "0.01"}}
    }



# import numpy as np

# def calculate_inertia_tensor_cubes(voxels, mass, H, voxel_side_length=1):
#     """
#     Calculate the inertia tensor of a rigid body represented by voxel cubes.
    
#     Args:
#     - voxels (numpy array): Array of voxel coordinates in shape (n, 3).
#     - mass (float): Mass of each voxel.
#     - H (numpy array): Homogeneous transformation matrix (4x4).
#     - voxel_side_length (float): Side length of each voxel cube.
    
#     Returns:
#     - numpy array: Inertia tensor (3x3 matrix).
#     """
#     # Apply the transformation to the voxel coordinates
#     transformed_voxels = (H @ np.hstack((voxels, np.ones((voxels.shape[0], 1)))).T).T[:, :3]
#     center_of_mass = np.mean(transformed_voxels, axis=0)
#     inertia_tensor = np.zeros((3, 3))

#     Rs = transformed_voxels - center_of_mass

#     # Vectorized computation of sum of R^2 * I and outer products
#     R_squares = np.sum(Rs**2, axis=1)
#     outer_Rs = np.einsum('ij,ik->ijk', Rs, Rs)
#     inertia_tensor = mass * (np.sum(R_squares[:, np.newaxis, np.newaxis] * np.eye(3), axis=0) - np.sum(outer_Rs, axis=0))

#     # Add the inertia tensor of each cube about its own center (mass moment of inertia of a cube)
#     inertia_contribution_from_self = (mass * voxel_side_length**2) / 6 * np.eye(3) * Rs.shape[0]
#     inertia_tensor += inertia_contribution_from_self

#     return inertia_tensor, center_of_mass

# inertial_matrix, CoM = calculate_inertia_tensor(voxels=np.array([[1,0,0],[1,0,1]]), mass=1, H=np.eye(4))
# print(inertial_matrix, CoM)
# inertial_matrix, CoM = calculate_inertia_tensor_cubes(voxels=np.array([[0,0,0],[0,0,1]]), mass=1, H=np.eye(4))
# print(inertial_matrix, CoM)