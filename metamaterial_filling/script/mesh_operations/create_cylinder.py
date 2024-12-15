import numpy as np
import trimesh

def create_cylinder(face_center, radius, normal_vector, height, segments=32):
    """
    Create a 3D cylinder mesh using the center point of one circular face, radius, 
    normal vector, and height. Ensures proper face orientation for CGAL.
    
    Parameters:
        face_center (array-like): The center point of the given circular face [x, y, z].
        radius (float): The radius of the circular face.
        normal_vector (array-like): The normal vector of the face pointing to the inside of the cylinder.
        height (float): The height of the cylinder.
        segments (int): Number of segments for approximating the circular face (more segments = smoother circle).
    
    Returns:
        trimesh.Trimesh: The generated cylinder mesh, with corrected face orientation.
    
    Raises:
        ValueError: If the radius is non-positive or the normal_vector has zero magnitude.
    """
    # Convert inputs to numpy arrays
    face_center = np.array(face_center)
    normal_vector = np.array(normal_vector)

    # Normalize the normal vector
    if np.linalg.norm(normal_vector) == 0:
        raise ValueError("The normal vector cannot be zero.")
    
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    if radius <= 0:
        raise ValueError("The radius must be positive.")

    # Calculate two perpendicular vectors to the normal vector
    # Use Gram-Schmidt process to generate a perpendicular vector
    arbitrary_vector = np.array([1, 0, 0]) if not np.allclose(normal_vector, [1, 0, 0]) else np.array([0, 1, 0])
    tangent_vector1 = np.cross(normal_vector, arbitrary_vector)
    tangent_vector1 = tangent_vector1 / np.linalg.norm(tangent_vector1)
    tangent_vector2 = np.cross(normal_vector, tangent_vector1)
    tangent_vector2 = tangent_vector2 / np.linalg.norm(tangent_vector2)

    # Calculate points for the base circle
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    circle_points = face_center + radius * (np.cos(theta)[:, None] * tangent_vector1 + np.sin(theta)[:, None] * tangent_vector2)
    
    # Calculate the offset to the opposite face using the normal vector and cylinder height
    offset = normal_vector * height

    # Calculate points for the opposite circle
    opposite_circle_points = circle_points + offset

    # Stack vertices together
    vertices = np.vstack((circle_points, opposite_circle_points))

    # Create faces for the side walls of the cylinder
    faces = []
    n = len(circle_points)

    for i in range(n):
        next_i = (i + 1) % n
        # Connect corresponding vertices on the two circles to form a quad split into 2 triangles
        faces.append([i, next_i, n + i])  # Triangle 1 for side wall
        faces.append([next_i, n + next_i, n + i])  # Triangle 2 for side wall

    # Create faces for the top and bottom circle caps
    center_top = face_center + offset
    center_bottom = face_center

    # Add the center points for the circular faces
    vertices = np.vstack((vertices, center_top, center_bottom))
    center_top_index = len(vertices) - 2
    center_bottom_index = len(vertices) - 1

    for i in range(n):
        next_i = (i + 1) % n
        # Top face
        faces.append([center_top_index, n + i, n + next_i])
        # Bottom face
        faces.append([center_bottom_index, i, next_i])

    # Create the cylinder mesh
    cylinder_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # cylinder_mesh.rezero()  # Center the mesh at the origin if needed
    cylinder_mesh.fix_normals()  # Fix normals if needed

    return cylinder_mesh


if __name__ == "__main__":
    # Example usage
    face_center = [0, 0, 0]
    radius = 1
    normal_vector = [1, 1, 1]
    height = 3

    cylinder_mesh = create_cylinder(face_center, radius, normal_vector, height)

    # Visualize the mesh
    cylinder_mesh.show()
