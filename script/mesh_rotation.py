import numpy as np
from stl import mesh
from scipy.spatial.transform import Rotation as R
import argparse

'''
This script rotates an STL mesh around an axis by a specified angle.
We assume the input stl mesh's z-axis is pointing up. If not, you may use the following code to rotate the mesh to the correct orientation.
'''
def rotate_mesh(input_stl_path, output_stl_path, angle_degrees, axis):
    # Load the STL file
    mesh_data = mesh.Mesh.from_file(input_stl_path)

    # Normalize the axis vector
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # Create the rotation matrix
    rotation = R.from_rotvec(np.radians(angle_degrees) * axis)
    
    # Reshape the mesh data to (N*3, 3) where N is the number of triangles
    vectors = mesh_data.vectors.reshape(-1, 3)

    # Apply the rotation to each point in the mesh
    rotated_vectors = rotation.apply(vectors)

    # Reshape the rotated vectors back to the original shape (N, 3, 3)
    mesh_data.vectors = rotated_vectors.reshape(-1, 3, 3)
    
    # Save the rotated mesh to the output path
    mesh_data.save(output_stl_path)

    print(f'Mesh rotated by {angle_degrees} degrees around axis {axis} and saved to {output_stl_path}')


if __name__ == '__main__':
    # Example usage
    # rotate_mesh('input.stl', 'rotated_output.stl', 90, [1, 0, 0])  # Rotate 90 degrees around the x-axis

    parser = argparse.ArgumentParser(description='Rotate an STL mesh around an axis by a specified angle.')
    parser.add_argument('--input_stl_path', type=str, help='Path to the input STL file')
    parser.add_argument('--output_stl_path', type=str, help='Path to save the rotated STL file')
    parser.add_argument('--angle_degrees', type=float, help='Angle to rotate the mesh in degrees')
    parser.add_argument('--axis', type=float, nargs=3, help='Axis to rotate around as a 3D vector. E.g. 1 0 0 for x-axis')

    args = parser.parse_args()
    rotate_mesh(args.input_stl_path, args.output_stl_path, args.angle_degrees, args.axis)

