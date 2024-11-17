import numpy as np
import trimesh

def create_box(face_center, face_length, face_width, normal_vector, width_direction, box_height):
    """
    Create a 3D box mesh using the center point of one face, face dimensions, normal vector,
    width direction vector, and height.
    
    Parameters:
        face_center (array-like): The center point of the given face [x, y, z].
        face_length (float): The length of the given face.
        face_width (float): The width of the given face.
        normal_vector (array-like): The normal vector of the face pointing to the inside of the box.
        width_direction (array-like): A vector pointing in the width direction of the face.
        box_height (float): The height of the box.
    
    Returns:
        trimesh.Trimesh: The generated box mesh.
    
    Raises:
        ValueError: If the width_direction and normal_vector are not orthogonal.
    """
    # Convert inputs to numpy arrays
    face_center = np.array(face_center)
    normal_vector = np.array(normal_vector)
    width_direction = np.array(width_direction)

    # Normalize the normal vector and width direction vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    width_direction = width_direction / np.linalg.norm(width_direction)

    # Check if the normal vector and width direction are orthogonal
    dot_product = np.dot(normal_vector, width_direction)
    if not np.isclose(dot_product, 0, atol=1e-6):
        raise ValueError("The width_direction and normal_vector must be orthogonal.")

    # Calculate the "right" vector (in the length direction) using cross product
    right_vector = np.cross(width_direction, normal_vector)
    right_vector = right_vector / np.linalg.norm(right_vector)
    
    # Ensure width_direction is perpendicular to normal_vector
    up_vector = np.cross(normal_vector, right_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)

    # Calculate half sizes
    half_length = face_length / 2
    half_width = face_width / 2

    # Calculate the 4 corner points of the initial face
    corner1 = face_center + half_length * right_vector + half_width * up_vector
    corner2 = face_center - half_length * right_vector + half_width * up_vector
    corner3 = face_center - half_length * right_vector - half_width * up_vector
    corner4 = face_center + half_length * right_vector - half_width * up_vector

    # Calculate the offset to the opposite face using the normal vector and box height
    offset = normal_vector * box_height

    # Calculate the 4 corner points of the opposite face
    corner5 = corner1 + offset
    corner6 = corner2 + offset
    corner7 = corner3 + offset
    corner8 = corner4 + offset

    # All vertices of the box
    vertices = np.array([corner1, corner2, corner3, corner4, corner5, corner6, corner7, corner8])

    # Define the 12 triangles (2 per face) to create the box mesh
    faces = [
        [0, 1, 2], [0, 2, 3],  # First face
        [4, 5, 6], [4, 6, 7],  # Opposite face
        [0, 1, 5], [0, 5, 4],  # Side face 1
        [1, 2, 6], [1, 6, 5],  # Side face 2
        [2, 3, 7], [2, 7, 6],  # Side face 3
        [3, 0, 4], [3, 4, 7]   # Side face 4
    ]

    # Create the box mesh
    box_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    return box_mesh

if __name__ == "__main__":
    # Example usage
    face_center = [0, 0, 0]
    face_length = 2
    face_width = 1
    normal_vector = [0, 0, -1]
    width_direction = [1, 1, 0]  # A vector pointing in the width direction of the face
    box_height = 3

    mesh = create_box(face_center, face_length, face_width, normal_vector, width_direction, box_height)

    # Save the mesh to an STL file
    mesh.export('box_mesh.stl')
