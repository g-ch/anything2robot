import numpy as np
import argparse
import os

def read_inner_points_bin(file_path='points.bin'):
    # Open the binary file for reading
    vector3d_size = 3 * 8  # 3 doubles, each 8 bytes
    with open(file_path, 'rb') as f:
        # Get the file size
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0, os.SEEK_SET)
        
        # Calculate the number of points
        num_points = file_size // vector3d_size
        
        # Read the points
        points = np.fromfile(f, dtype=np.float64, count=num_points * 3)
        
        # Reshape the points to a (num_points, 3) array
        points = points.reshape((num_points, 3))

    return points


if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Read the inner points from a binary file')
    parser.add_argument('--file_path', type=str, help='The path to the binary file containing the inner points')
    args = parser.parse_args()

    # Read the inner points
    points = read_inner_points_bin(args.file_path)

    # Print the number of points and the first 10 points
    print(f'Number of points: {points.shape[0]}')
    print('First 10 points:')
    print(points[:10])