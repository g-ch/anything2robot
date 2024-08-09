import open3d as o3d
import plotly.graph_objects as go  
import numpy as np

apply_transform = lambda point, transformation: (np.hstack((np.array(point).reshape(-1, 3), np.ones((np.array(point).reshape(-1, 3).shape[0], 1)))) @ transformation.T)[:,:3]

def get_voxel_faces(center, size):
    # Cube vertices and faces
    r = [-size / 2, size / 2]
    vertices = np.array([[x, y, z] for x in r for y in r for z in r]) + center
    faces = [
        [vertices[i] for i in [0, 1, 3, 2]], [vertices[i] for i in [4, 5, 7, 6]],
        [vertices[i] for i in [0, 1, 5, 4]], [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [0, 2, 6, 4]], [vertices[i] for i in [1, 3, 7, 5]]
    ]
    return np.array(faces)

def create_voxel_visualization(voxel_grid, voxel_size):
    # Extract voxel centers
    voxel_centers = np.asarray([voxel.grid_index for voxel in voxel_grid.get_voxels()])

    # Plot each voxel
    x, y, z, i, j, k = [], [], [], [], [], []
    for center in voxel_centers * voxel_size:
        voxel_faces = get_voxel_faces(center, voxel_size)
        for face in voxel_faces:
            start_idx = len(x)
            x.extend(face[:, 0])
            y.extend(face[:, 1])
            z.extend(face[:, 2])
            i.extend([start_idx, start_idx, start_idx+1, start_idx+2, start_idx+3, start_idx])
            j.extend([start_idx+1, start_idx+2, start_idx+3, start_idx, start_idx+1, start_idx+2])
            k.extend([start_idx+2, start_idx+3, start_idx, start_idx+1, start_idx+2, start_idx+3])

    # Create Plotly figure
    return go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, opacity=0.01, color='blue')

def create_joint_visualization(joint_dict):
    # Extract joints and their coordinates
    x_coords = []
    y_coords = []
    z_coords = []
    for joint, coords in joint_dict.items():
        x_coords.append(coords[0])
        y_coords.append(coords[1])
        z_coords.append(coords[2])

    # Create a scatter plot for the joints
    joints_scatter = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(size=5, color='#156082'),
        name='Joints'
    )

    # Create lines (hips to knees and knees to ankles)
    lines = [
        ("waist", "hip"),
        ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
        ('hip', 'left_hip'),('hip', 'right_hip'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
        ('waist', 'scapula'), ('scapula', 'left_shoulder'), ('scapula', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('left_wrist', 'left_hand'), ('right_wrist', 'right_hand'),
        ('left_ankle', 'left_foot'), ('right_ankle', 'right_foot')
    ]

    line_data = []
    for start, end in lines:
        line_data.extend([
            go.Scatter3d(
                x=[joint_dict[start][0], joint_dict[end][0]],
                y=[joint_dict[start][1], joint_dict[end][1]],
                z=[joint_dict[start][2], joint_dict[end][2]],
                mode='lines',
                line=dict(color='black', width=5),
                name=f'{start} to {end}'
            )
        ])

    return joints_scatter, line_data

def create_mesh(mesh_file):
    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    # Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    return go.Mesh3d(x=vertices[:, 0],
                                y=vertices[:, 1],
                                z=vertices[:, 2],
                                i=triangles[:, 0],
                                j=triangles[:, 1],
                                k=triangles[:, 2],
                                opacity=0.1,
                                color='grey')

def create_transformed_mesh(mesh_file, new_origin, new_x_axis, new_y_axis, new_z_axis, expected_x=11):
    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    # Extract vertices and triangles
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Create rotation matrix from the new axes
    rotation_matrix = np.column_stack([new_x_axis, new_y_axis, new_z_axis])

    # Apply rotation
    transformed_vertices = (vertices - new_origin) @ rotation_matrix
    max_x = np.max(transformed_vertices[:, 0])
    scale = expected_x / max_x

    # Create Plotly Mesh3d object with transformed vertices
    return go.Mesh3d(
        x=scale * transformed_vertices[:, 0],
        y=scale * transformed_vertices[:, 1],
        z=scale * transformed_vertices[:, 2],
        i=triangles[:, 0],
        j=triangles[:, 1],
        k=triangles[:, 2],
        opacity=0.2,
        color='grey'
    ), scale

def create_cylinder_surface(height, radius, center=(0, 0, 0), n_points=50):
    # Create a cylinder
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(0, height, n_points)
    theta, z = np.meshgrid(theta, z)
    x = radius * np.cos(theta) + center[0]
    y = radius * np.sin(theta) + center[1]
    z = z + center[2]

    # Create the surface data for the cylinder
    cylinder_surface = go.Surface(x=x, y=y, z=z, colorscale='Blues')

    return cylinder_surface

# Rotation matrix to align with the main axis
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        return np.eye(3)
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def create_ellipsoid(point1, point2, axis1=1, axis2=1):

    center = (point1 + point2) / 2
    direction = point2 - point1
    rot_matrix = rotation_matrix_from_vectors(np.array([1, 0, 0]), direction)
    
    a = np.linalg.norm(point2 - point1)  # Semi-major axis length
    b = axis1  # Second semi-axis
    c = axis2  # Third semi-axis
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # Apply rotation and translation (same as before)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            [x[i, j], y[i, j], z[i, j]] = np.dot(rot_matrix, [x[i, j], y[i, j], z[i, j]]) + center

    return go.Surface(x=x, y=y, z=z, opacity=0.5, showscale=False)

def create_cylinder(center, height, radius, axis='z', n_points=50):
    """
    Create a cylinder in Plotly.

    Args:
    - center (array_like): The center of the base of the cylinder.
    - height (float): The height of the cylinder.
    - radius (float): The radius of the cylinder.
    - axis (str): The axis along which the cylinder is oriented ('x', 'y', or 'z').
    - n_points (int): Number of points for approximation.

    Returns:
    - plotly.graph_objects.Surface: A Plotly Surface object representing the cylinder.
    """
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, height, n_points)
    u, v = np.meshgrid(u, v)
    
    if axis == 'x':
        x = v + center[0]
        y = radius * np.cos(u) + center[1]
        z = radius * np.sin(u) + center[2]
    elif axis == 'y':
        x = radius * np.cos(u) + center[0]
        y = v + center[1]
        z = radius * np.sin(u) + center[2]
    else:  # default to 'z' axis
        x = radius * np.cos(u) + center[0]
        y = radius * np.sin(u) + center[1]
        z = v + center[2]

    return go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale='Blues', showscale=False)

# Function to create lines representing coordinate axes
def create_axes_lines(origin, x_axis, y_axis, z_axis, length=10):
    axes_lines = []
    # X-axis line (red)
    axes_lines.append(go.Scatter3d(x=[origin[0], origin[0] + length * x_axis[0]],
                                   y=[origin[1], origin[1] + length * x_axis[1]],
                                   z=[origin[2], origin[2] + length * x_axis[2]],
                                   mode='lines', line=dict(color='red', width=2)))
    # Y-axis line (green)
    axes_lines.append(go.Scatter3d(x=[origin[0], origin[0] + length * y_axis[0]],
                                   y=[origin[1], origin[1] + length * y_axis[1]],
                                   z=[origin[2], origin[2] + length * y_axis[2]],
                                   mode='lines', line=dict(color='green', width=2)))
    # Z-axis line (blue)
    axes_lines.append(go.Scatter3d(x=[origin[0], origin[0] + length * z_axis[0]],
                                   y=[origin[1], origin[1] + length * z_axis[1]],
                                   z=[origin[2], origin[2] + length * z_axis[2]],
                                   mode='lines', line=dict(color='blue', width=2)))

    return axes_lines

# Initialize a new VoxelGrid for transformed voxels
def create_voxel_grid_np(grid, voxel_size):
    x, y, z = np.array(grid).T
    scatter = go.Scatter3d(
        x=x, y=y, z=z, 
        mode='markers', 
        marker=dict(size=voxel_size, opacity=0.5)
    )
    return scatter
